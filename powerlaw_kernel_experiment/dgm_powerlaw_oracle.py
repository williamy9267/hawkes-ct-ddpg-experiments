from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from powerlaw_env import PowerLawHawkesConfig, PowerLawHawkesEnv


@dataclass
class DGMConfig:
    hidden_dim: int = 80
    num_layers: int = 3
    lr: float = 2e-3
    steps: int = 600
    batch_size: int = 384
    action_grid_size: int = 41
    x_min: float = -1.2
    x_max: float = 3.0
    z_min: float = 0.0
    z_max: float = 8.0
    action_fd_eps: float = 0.02
    residual_weight: float = 1.0
    positivity_weight: float = 1e-3
    grad_clip: float = 5.0
    seed: int = 123
    device: str = "cuda"
    print_every: int = 100
    sampling_mode: str = "reachable"
    reachable_episodes: int = 100
    reachable_action_noise: float = 0.20
    reachable_seed: int = 9021
    reachable_jitter_t: float = 0.02
    reachable_jitter_x: float = 0.04
    reachable_jitter_z: float = 0.03
    uniform_sample_fraction: float = 0.12
    validation_every: int = 0
    validation_episodes: int = 0
    validation_seed: int = 7123
    early_stop_patience: int = 3
    restore_best: bool = True


class DGMLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.g = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.h = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        xs = torch.cat([x, s], dim=1)
        z = torch.sigmoid(self.z(xs))
        g = torch.sigmoid(self.g(xs))
        r = torch.sigmoid(self.r(xs))
        xsr = torch.cat([x, s * r], dim=1)
        h = torch.tanh(self.h(xsr))
        return (1.0 - g) * h + z * s


class DGMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([DGMLayer(input_dim, hidden_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.tanh(self.input_layer(x))
        for layer in self.layers:
            s = layer(x, s)
        return self.output_layer(s)


def control_effectiveness_torch(actions: torch.Tensor, cfg: PowerLawHawkesConfig) -> torch.Tensor:
    return actions / (cfg.action_half_saturation + actions + 1e-12)


def readout_multiplier_torch(actions: torch.Tensor, cfg: PowerLawHawkesConfig) -> torch.Tensor:
    return 1.0 - cfg.control_effect_strength * control_effectiveness_torch(actions, cfg)


def kernel_readout_torch(z: torch.Tensor, cfg: PowerLawHawkesConfig) -> torch.Tensor:
    weights = torch.as_tensor(cfg.weights, dtype=z.dtype, device=z.device).view(1, -1)
    return torch.sum(z * weights, dim=1, keepdim=True)


def intensity_torch(x: torch.Tensor, z: torch.Tensor, actions: torch.Tensor, cfg: PowerLawHawkesConfig) -> torch.Tensor:
    affine_baseline = cfg.mu0 + cfg.mu_x * x + cfg.mu_a * actions
    baseline = torch.clamp(affine_baseline, min=cfg.baseline_min, max=cfg.baseline_max)
    memory = cfg.excitation * readout_multiplier_torch(actions, cfg) * kernel_readout_torch(z, cfg)
    return baseline + torch.relu(memory)


def state_drift_torch(x: torch.Tensor, actions: torch.Tensor, cfg: PowerLawHawkesConfig) -> torch.Tensor:
    return cfg.drift_constant - cfg.drift_reversion * x - cfg.control_drift_strength * actions


def state_volatility_torch(x: torch.Tensor, actions: torch.Tensor, cfg: PowerLawHawkesConfig) -> torch.Tensor:
    return torch.clamp(
        cfg.diffusion + cfg.volatility_x_strength * x + cfg.control_volatility_strength * actions,
        min=1e-6,
    )


def jump_amplitude_torch(x: torch.Tensor, actions: torch.Tensor, cfg: PowerLawHawkesConfig) -> torch.Tensor:
    raw = cfg.jump_size + cfg.jump_size_x_strength * x + cfg.jump_size_action_strength * actions
    return torch.clamp(raw, min=cfg.jump_size_min, max=cfg.jump_size_max)


class DGMPowerLawOracle(nn.Module):
    """DGM/HJB oracle for the exponential-mixture Markov approximation."""

    def __init__(self, env_cfg: PowerLawHawkesConfig, dgm_cfg: DGMConfig):
        super().__init__()
        self.env_cfg = env_cfg
        self.dgm_cfg = dgm_cfg
        self.net = DGMNet(2 + len(env_cfg.filter_betas), dgm_cfg.hidden_dim, dgm_cfg.num_layers)

    def scaled_inputs(self, t: torch.Tensor, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        cfg = self.env_cfg
        dcfg = self.dgm_cfg
        t_s = 2.0 * t / cfg.horizon - 1.0
        x_s = 2.0 * (x - dcfg.x_min) / (dcfg.x_max - dcfg.x_min) - 1.0
        z_s = 2.0 * (z - dcfg.z_min) / (dcfg.z_max - dcfg.z_min) - 1.0
        return torch.cat([t_s, x_s, z_s], dim=1)

    def forward(self, t: torch.Tensor, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raw = self.net(self.scaled_inputs(t, x, z))
        terminal = self.env_cfg.terminal_x_cost * x * x
        return terminal + (self.env_cfg.horizon - t) * raw

    def value_np(self, times: np.ndarray, states: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            t = torch.as_tensor(times, dtype=torch.float32, device=device).reshape(-1, 1)
            x = torch.as_tensor(states[:, 0], dtype=torch.float32, device=device).reshape(-1, 1)
            z = torch.as_tensor(states[:, 1:], dtype=torch.float32, device=device)
            return self(t, x, z).detach().cpu().numpy().reshape(-1)

    def action_np(self, times: np.ndarray, states: np.ndarray) -> np.ndarray:
        self.eval()
        cfg = self.env_cfg
        dcfg = self.dgm_cfg
        with torch.no_grad():
            device = next(self.parameters()).device
            t = torch.as_tensor(times, dtype=torch.float32, device=device).reshape(-1, 1)
            x = torch.as_tensor(states[:, 0], dtype=torch.float32, device=device).reshape(-1, 1)
            z = torch.as_tensor(states[:, 1:], dtype=torch.float32, device=device)
            eps = dcfg.action_fd_eps
            u = self(t, x, z)
            u_plus = self(t, x + eps, z)
            u_minus = self(t, x - eps, z)
            u_x = (u_plus - u_minus) / (2.0 * eps)
            u_xx = (u_plus - 2.0 * u + u_minus) / (eps * eps)

            actions = torch.linspace(0.0, 1.0, dcfg.action_grid_size, device=device).view(1, -1)
            lam = intensity_torch(x, z, actions, cfg)
            jump_delta = jump_value_increment(self, t, x, z, actions)
            drift = state_drift_torch(x, actions, cfg)
            sigma = state_volatility_torch(x, actions, cfg)
            objective = cfg.control_cost * actions * actions + drift * u_x + 0.5 * sigma * sigma * u_xx + lam * jump_delta
            best = torch.argmin(objective, dim=1)
            return actions[0, best].detach().cpu().numpy()


class DGMOraclePolicy:
    def __init__(self, oracle: DGMPowerLawOracle):
        self.oracle = oracle

    def act(self, time: float, state: np.ndarray) -> float:
        actions = self.oracle.action_np(np.asarray([time]), np.asarray([state], dtype=np.float64))
        return float(actions[0])

    def value_path(self, times: np.ndarray, states: np.ndarray) -> np.ndarray:
        return self.oracle.value_np(times, states)


def jump_value_increment(
    model: DGMPowerLawOracle,
    t: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    cfg = model.env_cfg
    batch = t.shape[0]
    grid = actions.shape[1]
    t_grid = t.view(batch, 1).expand(batch, grid).reshape(-1, 1)
    x_grid = x.view(batch, 1).expand(batch, grid)
    gamma = jump_amplitude_torch(x, actions, cfg)
    x_jump = (x_grid + gamma).reshape(-1, 1)
    z_jump = z.view(batch, 1, -1).expand(batch, grid, z.shape[1]).reshape(batch * grid, z.shape[1]) + 1.0
    u_jump = model(t_grid, x_jump, z_jump).reshape(batch, grid)
    u_current = model(t, x, z).view(batch, 1)
    return u_jump - u_current


def _sample_uniform_domain(
    dgm_cfg: DGMConfig,
    env_cfg: PowerLawHawkesConfig,
    batch_size: int,
    device: torch.device,
):
    k = len(env_cfg.filter_betas)
    t = torch.rand(batch_size, 1, device=device) * env_cfg.horizon
    x = dgm_cfg.x_min + torch.rand(batch_size, 1, device=device) * (dgm_cfg.x_max - dgm_cfg.x_min)
    z = dgm_cfg.z_min + torch.rand(batch_size, k, device=device) * (dgm_cfg.z_max - dgm_cfg.z_min)
    return t, x, z


def build_reachable_state_bank(
    env_cfg: PowerLawHawkesConfig,
    dgm_cfg: DGMConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate on-model states for the high-dimensional memory coordinates.

    The power-law approximation has many exponential filters. Independent
    uniform samples in those coordinates are mostly unreachable and can dominate
    the PINN residual, especially with signed mixture weights. This bank keeps
    the DGM collocation points close to the states used by policy evaluation.
    """

    rng = np.random.default_rng(dgm_cfg.reachable_seed)
    times: list[float] = []
    states: list[np.ndarray] = []
    anchors = (0.15, 0.30, 0.45, 0.60)
    for episode_idx in range(max(1, dgm_cfg.reachable_episodes)):
        env = PowerLawHawkesEnv(env_cfg)
        state = env.reset()
        mode = episode_idx % (len(anchors) + 1)
        for _ in range(env_cfg.steps):
            times.append(float(env.t))
            states.append(state.copy())
            if mode == 0:
                action = rng.uniform(0.0, 1.0)
            else:
                action = anchors[mode - 1] + rng.normal(0.0, dgm_cfg.reachable_action_noise)
            state, _, done, _ = env.step(float(np.clip(action, 0.0, 1.0)), rng)
            if done:
                break
    return np.asarray(times, dtype=np.float32), np.asarray(states, dtype=np.float32)


def _sample_reachable_domain(
    dgm_cfg: DGMConfig,
    env_cfg: PowerLawHawkesConfig,
    batch_size: int,
    device: torch.device,
    state_bank: tuple[torch.Tensor, torch.Tensor],
):
    bank_times, bank_states = state_bank
    idx = torch.randint(0, bank_times.shape[0], (batch_size,), device=device)
    t = bank_times[idx].clone()
    state = bank_states[idx].clone()
    x = state[:, 0:1]
    z = state[:, 1:]
    if dgm_cfg.reachable_jitter_t > 0.0:
        t = t + dgm_cfg.reachable_jitter_t * (2.0 * torch.rand_like(t) - 1.0)
        t = torch.clamp(t, min=0.0, max=env_cfg.horizon)
    if dgm_cfg.reachable_jitter_x > 0.0:
        x = x + dgm_cfg.reachable_jitter_x * torch.randn_like(x)
    if dgm_cfg.reachable_jitter_z > 0.0:
        z = z + dgm_cfg.reachable_jitter_z * torch.randn_like(z)
        z = torch.clamp(z, min=0.0)
    return t, x, z


def sample_domain(
    dgm_cfg: DGMConfig,
    env_cfg: PowerLawHawkesConfig,
    batch_size: int,
    device: torch.device,
    state_bank: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    use_reachable = state_bank is not None and dgm_cfg.sampling_mode == "reachable"
    if not use_reachable:
        t, x, z = _sample_uniform_domain(dgm_cfg, env_cfg, batch_size, device)
    else:
        uniform_n = int(round(batch_size * max(0.0, min(1.0, dgm_cfg.uniform_sample_fraction))))
        reachable_n = batch_size - uniform_n
        t_parts = []
        x_parts = []
        z_parts = []
        if reachable_n > 0:
            t_r, x_r, z_r = _sample_reachable_domain(dgm_cfg, env_cfg, reachable_n, device, state_bank)
            t_parts.append(t_r)
            x_parts.append(x_r)
            z_parts.append(z_r)
        if uniform_n > 0:
            t_u, x_u, z_u = _sample_uniform_domain(dgm_cfg, env_cfg, uniform_n, device)
            t_parts.append(t_u)
            x_parts.append(x_u)
            z_parts.append(z_u)
        t = torch.cat(t_parts, dim=0)
        x = torch.cat(x_parts, dim=0)
        z = torch.cat(z_parts, dim=0)
        perm = torch.randperm(t.shape[0], device=device)
        t = t[perm]
        x = x[perm]
        z = z[perm]
    t.requires_grad_(True)
    x.requires_grad_(True)
    z.requires_grad_(True)
    return t, x, z


def hjb_residual(model: DGMPowerLawOracle, t: torch.Tensor, x: torch.Tensor, z: torch.Tensor):
    cfg = model.env_cfg
    dcfg = model.dgm_cfg
    u = model(t, x, z)
    u_t, u_x, u_z = torch.autograd.grad(u.sum(), (t, x, z), create_graph=True)
    (u_xx,) = torch.autograd.grad(u_x.sum(), x, create_graph=True)

    actions = torch.linspace(0.0, 1.0, dcfg.action_grid_size, device=t.device).view(1, -1)
    lam = intensity_torch(x, z, actions, cfg)
    jump_delta = jump_value_increment(model, t, x, z, actions)
    drift = state_drift_torch(x, actions, cfg)
    sigma = state_volatility_torch(x, actions, cfg)
    action_objective = cfg.control_cost * actions * actions + drift * u_x + 0.5 * sigma * sigma * u_xx + lam * jump_delta
    min_action_term, action_idx = torch.min(action_objective, dim=1, keepdim=True)

    betas = torch.as_tensor(cfg.filter_betas, dtype=z.dtype, device=z.device).view(1, -1)
    z_drift_term = torch.sum(-betas * z * u_z, dim=1, keepdim=True)
    residual = u_t + z_drift_term - cfg.beta * u + cfg.state_cost * x * x + min_action_term
    best_actions = actions[0, action_idx.reshape(-1)].reshape(-1, 1)
    return residual, best_actions, u


def train_dgm_oracle(
    env_cfg: PowerLawHawkesConfig | None = None,
    dgm_cfg: DGMConfig | None = None,
) -> Tuple[DGMPowerLawOracle, List[dict]]:
    env_cfg = env_cfg or PowerLawHawkesConfig()
    dgm_cfg = dgm_cfg or DGMConfig()
    if dgm_cfg.device == "cuda" and not torch.cuda.is_available():
        dgm_cfg = DGMConfig(**{**dgm_cfg.__dict__, "device": "cpu"})
    torch.manual_seed(dgm_cfg.seed)
    np.random.seed(dgm_cfg.seed)
    device = torch.device(dgm_cfg.device)
    model = DGMPowerLawOracle(env_cfg, dgm_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=dgm_cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, dgm_cfg.steps // 3), gamma=0.55)
    state_bank: tuple[torch.Tensor, torch.Tensor] | None = None
    if dgm_cfg.sampling_mode == "reachable":
        bank_times_np, bank_states_np = build_reachable_state_bank(env_cfg, dgm_cfg)
        state_bank = (
            torch.as_tensor(bank_times_np, dtype=torch.float32, device=device).reshape(-1, 1),
            torch.as_tensor(bank_states_np, dtype=torch.float32, device=device),
        )
        z_bank = bank_states_np[:, 1:]
        print(
            "dgm_reachable_bank "
            f"states={bank_states_np.shape[0]} "
            f"x_range=({bank_states_np[:, 0].min():.3f},{bank_states_np[:, 0].max():.3f}) "
            f"z_mean={z_bank.mean():.3f} z_p95={np.quantile(z_bank, 0.95):.3f} z_max={z_bank.max():.3f}"
        )

    history: List[dict] = []
    best_eval_return = -np.inf
    best_state = None
    evaluations_without_improvement = 0
    for step in range(1, dgm_cfg.steps + 1):
        t, x, z = sample_domain(dgm_cfg, env_cfg, dgm_cfg.batch_size, device, state_bank=state_bank)
        residual, best_actions, u = hjb_residual(model, t, x, z)
        residual_loss = torch.mean(residual * residual)
        positivity_loss = torch.mean(torch.relu(-u) ** 2)
        loss = dgm_cfg.residual_weight * residual_loss + dgm_cfg.positivity_weight * positivity_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), dgm_cfg.grad_clip)
        opt.step()
        scheduler.step()

        should_log = step == 1 or step % dgm_cfg.print_every == 0 or step == dgm_cfg.steps
        should_validate = (
            dgm_cfg.validation_episodes > 0
            and dgm_cfg.validation_every > 0
            and (step == 1 or step % dgm_cfg.validation_every == 0 or step == dgm_cfg.steps)
        )
        eval_stats = None
        if should_validate:
            eval_stats = evaluate_dgm_policy(
                env_cfg,
                model,
                seed=dgm_cfg.validation_seed,
                episodes=dgm_cfg.validation_episodes,
            )
            if eval_stats["return_mean"] > best_eval_return:
                best_eval_return = eval_stats["return_mean"]
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1

        if should_log or should_validate:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "residual_loss": float(residual_loss.detach().cpu()),
                "positivity_loss": float(positivity_loss.detach().cpu()),
                "mean_action": float(best_actions.detach().mean().cpu()),
                "lr": float(scheduler.get_last_lr()[0]),
                "device": str(device),
                "sampling_mode": dgm_cfg.sampling_mode,
            }
            if eval_stats is not None:
                row.update({f"eval_{key}": value for key, value in eval_stats.items()})
                row["is_best_checkpoint"] = float(eval_stats["return_mean"] >= best_eval_return)
            history.append(row)
            message = (
                "dgm_step={step:04d} loss={loss:9.5f} residual={residual_loss:9.5f} "
                "mean_action={mean_action:5.3f} lr={lr:.2e} device={device}".format(**row)
            )
            if eval_stats is not None:
                message += (
                    f" eval_return={eval_stats['return_mean']:.4f} "
                    f"eval_action={eval_stats['action_mean']:.3f}"
                )
            print(message)
        if (
            should_validate
            and dgm_cfg.early_stop_patience > 0
            and evaluations_without_improvement >= dgm_cfg.early_stop_patience
        ):
            print(
                f"dgm early stopping at step {step}; no validation improvement for "
                f"{evaluations_without_improvement} evaluations"
            )
            break
    if dgm_cfg.restore_best and best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    model.eval()
    return model, history


def save_dgm_oracle(
    path: str | Path,
    model: DGMPowerLawOracle,
    history: List[dict],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "environment": asdict(model.env_cfg),
            "dgm_config": asdict(model.dgm_cfg),
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "history": history,
        },
        target,
    )


def load_dgm_oracle(
    path: str | Path,
    device: str | None = None,
) -> Tuple[DGMPowerLawOracle, List[dict]]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    env_cfg = PowerLawHawkesConfig(**checkpoint["environment"])
    dgm_values = dict(checkpoint["dgm_config"])
    requested_device = device or dgm_values.get("device", "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    dgm_values["device"] = requested_device
    dgm_cfg = DGMConfig(**dgm_values)
    model = DGMPowerLawOracle(env_cfg, dgm_cfg).to(torch.device(requested_device))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, list(checkpoint.get("history", []))


def evaluate_dgm_policy(
    env_cfg: PowerLawHawkesConfig,
    oracle: DGMPowerLawOracle,
    seed: int,
    episodes: int,
) -> dict:
    """Evaluate all DGM actions in one neural-network batch per time step."""

    envs = [PowerLawHawkesEnv(env_cfg) for _ in range(episodes)]
    states = np.stack([env.reset() for env in envs])
    rngs = [np.random.default_rng(seed + 10_000 * idx) for idx in range(episodes)]
    returns = np.zeros(episodes, dtype=np.float64)
    action_sums = np.zeros(episodes, dtype=np.float64)
    jump_sums = np.zeros(episodes, dtype=np.float64)
    intensity_sums = np.zeros(episodes, dtype=np.float64)
    all_actions = []
    for k in range(env_cfg.steps):
        time = k * env_cfg.dt
        actions = oracle.action_np(np.full(episodes, time), states)
        next_states = []
        for idx, env in enumerate(envs):
            state, reward_rate, _, info = env.step(float(actions[idx]), rngs[idx])
            next_states.append(state)
            returns[idx] += np.exp(-env_cfg.beta * time) * reward_rate * env_cfg.dt
            action_sums[idx] += actions[idx]
            jump_sums[idx] += info["jumps"]
            intensity_sums[idx] += info["lambda"]
        all_actions.extend(actions.tolist())
        states = np.stack(next_states)
    terminal = PowerLawHawkesEnv(env_cfg).terminal_reward(states)
    returns += np.exp(-env_cfg.beta * env_cfg.horizon) * terminal
    action_arr = np.asarray(all_actions, dtype=np.float64)
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_se": float(np.std(returns) / np.sqrt(max(1, episodes))),
        "return_ci90": float(1.6448536269514722 * np.std(returns) / np.sqrt(max(1, episodes))),
        "action_mean": float(np.mean(action_sums / env_cfg.steps)),
        "action_std": float(np.std(action_arr)),
        "action_min": float(np.min(action_arr)),
        "action_max": float(np.max(action_arr)),
        "jumps_mean": float(np.mean(jump_sums)),
        "intensity_mean": float(np.mean(intensity_sums / env_cfg.steps)),
        "episodes": int(episodes),
    }
