from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from erlang_env import ErlangHawkesConfig, ErlangHawkesEnv


@dataclass
class DGMConfig:
    hidden_dim: int = 72
    num_layers: int = 3
    lr: float = 2e-3
    steps: int = 450
    batch_size: int = 384
    action_grid_size: int = 41
    x_min: float = -1.2
    x_max: float = 3.0
    y_min: float = 0.0
    y_max: float = 8.0
    action_fd_eps: float = 0.02
    residual_weight: float = 1.0
    positivity_weight: float = 1e-3
    grad_clip: float = 5.0
    seed: int = 123
    device: str = "cuda"
    print_every: int = 100


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


def control_effectiveness_torch(actions: torch.Tensor, cfg: ErlangHawkesConfig) -> torch.Tensor:
    return actions / (cfg.action_half_saturation + actions + 1e-12)


def readout_multiplier_torch(actions: torch.Tensor, cfg: ErlangHawkesConfig) -> torch.Tensor:
    return 1.0 - cfg.control_effect_strength * control_effectiveness_torch(actions, cfg)


def intensity_torch(x: torch.Tensor, y2: torch.Tensor, actions: torch.Tensor, cfg: ErlangHawkesConfig) -> torch.Tensor:
    affine_baseline = cfg.mu0 + cfg.mu_x * x + cfg.mu_a * actions
    baseline = torch.clamp(affine_baseline, min=cfg.baseline_min, max=cfg.baseline_max)
    memory = cfg.excitation * readout_multiplier_torch(actions, cfg) * y2
    return baseline + memory


def state_drift_torch(x: torch.Tensor, actions: torch.Tensor, cfg: ErlangHawkesConfig) -> torch.Tensor:
    return cfg.drift_constant - cfg.drift_reversion * x - cfg.control_drift_strength * actions


def state_volatility_torch(x: torch.Tensor, actions: torch.Tensor, cfg: ErlangHawkesConfig) -> torch.Tensor:
    return torch.clamp(
        cfg.diffusion + cfg.volatility_x_strength * x + cfg.control_volatility_strength * actions,
        min=1e-6,
    )


def jump_amplitude_torch(x: torch.Tensor, actions: torch.Tensor, cfg: ErlangHawkesConfig) -> torch.Tensor:
    raw = cfg.jump_size + cfg.jump_size_x_strength * x + cfg.jump_size_action_strength * actions
    return torch.clamp(raw, min=cfg.jump_size_min, max=cfg.jump_size_max)


class DGMErlangOracle(nn.Module):
    """Known-parameter DGM/HJB oracle in the exact Erlang state (t, X, Y1, Y2)."""

    def __init__(self, env_cfg: ErlangHawkesConfig, dgm_cfg: DGMConfig):
        super().__init__()
        self.env_cfg = env_cfg
        self.dgm_cfg = dgm_cfg
        self.net = DGMNet(4, dgm_cfg.hidden_dim, dgm_cfg.num_layers)

    def scaled_inputs(self, t: torch.Tensor, x: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        cfg = self.env_cfg
        dcfg = self.dgm_cfg
        t_s = 2.0 * t / cfg.horizon - 1.0
        x_s = 2.0 * (x - dcfg.x_min) / (dcfg.x_max - dcfg.x_min) - 1.0
        y1_s = 2.0 * (y1 - dcfg.y_min) / (dcfg.y_max - dcfg.y_min) - 1.0
        y2_s = 2.0 * (y2 - dcfg.y_min) / (dcfg.y_max - dcfg.y_min) - 1.0
        return torch.cat([t_s, x_s, y1_s, y2_s], dim=1)

    def forward(self, t: torch.Tensor, x: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        raw = self.net(self.scaled_inputs(t, x, y1, y2))
        terminal = self.env_cfg.terminal_x_cost * x * x
        return terminal + (self.env_cfg.horizon - t) * raw

    def value_np(self, times: np.ndarray, states: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            t = torch.as_tensor(times, dtype=torch.float32, device=device).reshape(-1, 1)
            x = torch.as_tensor(states[:, 0], dtype=torch.float32, device=device).reshape(-1, 1)
            y1 = torch.as_tensor(states[:, 1], dtype=torch.float32, device=device).reshape(-1, 1)
            y2 = torch.as_tensor(states[:, 2], dtype=torch.float32, device=device).reshape(-1, 1)
            return self(t, x, y1, y2).detach().cpu().numpy().reshape(-1)

    def action_np(self, times: np.ndarray, states: np.ndarray) -> np.ndarray:
        self.eval()
        cfg = self.env_cfg
        dcfg = self.dgm_cfg
        with torch.no_grad():
            device = next(self.parameters()).device
            t = torch.as_tensor(times, dtype=torch.float32, device=device).reshape(-1, 1)
            x = torch.as_tensor(states[:, 0], dtype=torch.float32, device=device).reshape(-1, 1)
            y1 = torch.as_tensor(states[:, 1], dtype=torch.float32, device=device).reshape(-1, 1)
            y2 = torch.as_tensor(states[:, 2], dtype=torch.float32, device=device).reshape(-1, 1)
            eps = dcfg.action_fd_eps
            u = self(t, x, y1, y2)
            u_plus = self(t, x + eps, y1, y2)
            u_minus = self(t, x - eps, y1, y2)
            u_x = (u_plus - u_minus) / (2.0 * eps)
            u_xx = (u_plus - 2.0 * u + u_minus) / (eps * eps)

            actions = torch.linspace(0.0, 1.0, dcfg.action_grid_size, device=device).view(1, -1)
            lam = intensity_torch(x, y2, actions, cfg)
            jump_delta = jump_value_increment(self, t, x, y1, y2, actions)
            drift = state_drift_torch(x, actions, cfg)
            sigma = state_volatility_torch(x, actions, cfg)
            objective = cfg.control_cost * actions * actions + drift * u_x + 0.5 * sigma * sigma * u_xx + lam * jump_delta
            best = torch.argmin(objective, dim=1)
            return actions[0, best].detach().cpu().numpy()


class DGMOraclePolicy:
    def __init__(self, oracle: DGMErlangOracle):
        self.oracle = oracle

    def act(self, time: float, state: np.ndarray) -> float:
        actions = self.oracle.action_np(np.asarray([time]), np.asarray([state], dtype=np.float64))
        return float(actions[0])

    def value_path(self, times: np.ndarray, states: np.ndarray) -> np.ndarray:
        return self.oracle.value_np(times, states)


def evaluate_dgm_policy(
    env_cfg: ErlangHawkesConfig,
    policy: DGMOraclePolicy,
    seed: int,
    episodes: int,
) -> dict:
    """Evaluate DGM paths with one batched GPU policy call per time step."""

    envs = [ErlangHawkesEnv(env_cfg) for _ in range(episodes)]
    rngs = [np.random.default_rng(seed + 10_000 * idx) for idx in range(episodes)]
    states = np.stack([env.reset() for env in envs])
    returns = np.zeros(episodes, dtype=np.float64)
    action_sums = np.zeros(episodes, dtype=np.float64)
    jump_sums = np.zeros(episodes, dtype=np.float64)
    intensity_sums = np.zeros(episodes, dtype=np.float64)
    for step in range(env_cfg.steps):
        time = step * env_cfg.dt
        actions = policy.oracle.action_np(np.full(episodes, time), states)
        next_states = []
        for idx, (env, rng) in enumerate(zip(envs, rngs)):
            next_state, reward_rate, done, info = env.step(float(actions[idx]), rng)
            returns[idx] += np.exp(-env_cfg.beta * time) * reward_rate * env_cfg.dt
            action_sums[idx] += float(actions[idx])
            jump_sums[idx] += float(info["jumps"])
            intensity_sums[idx] += float(info["lambda"])
            next_states.append(next_state)
        states = np.stack(next_states)
    terminal = np.asarray(
        [env.terminal_reward(np.asarray([state]))[0] for env, state in zip(envs, states)],
        dtype=np.float64,
    )
    returns += np.exp(-env_cfg.beta * env_cfg.horizon) * terminal
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_se": float(np.std(returns) / np.sqrt(max(1, episodes))),
        "return_ci90": float(1.6448536269514722 * np.std(returns) / np.sqrt(max(1, episodes))),
        "action_mean": float(np.mean(action_sums / env_cfg.steps)),
        "jumps_mean": float(np.mean(jump_sums)),
        "intensity_mean": float(np.mean(intensity_sums / env_cfg.steps)),
        "episodes": int(episodes),
    }


def save_dgm_oracle(model: DGMErlangOracle, path: str | Path) -> None:
    from dataclasses import asdict

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "environment": asdict(model.env_cfg),
            "config": asdict(model.dgm_cfg),
        },
        output,
    )


def load_dgm_oracle(
    path: str | Path,
    env_cfg: ErlangHawkesConfig | None = None,
    device: str | None = None,
) -> DGMErlangOracle:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    stored_cfg = checkpoint.get("config", {})
    selected_device = device or stored_cfg.get("device", "cpu")
    if selected_device == "cuda" and not torch.cuda.is_available():
        selected_device = "cpu"
    dgm_cfg = DGMConfig(**{**stored_cfg, "device": selected_device})
    stored_environment = checkpoint.get("environment", checkpoint.get("env"))
    if env_cfg is None and stored_environment is None:
        raise ValueError("DGM checkpoint does not contain an environment configuration")
    resolved_env = env_cfg or ErlangHawkesConfig(**stored_environment)
    model = DGMErlangOracle(resolved_env, dgm_cfg).to(selected_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def jump_value_increment(
    model: DGMErlangOracle,
    t: torch.Tensor,
    x: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    cfg = model.env_cfg
    batch = t.shape[0]
    grid = actions.shape[1]
    t_grid = t.view(batch, 1).expand(batch, grid).reshape(-1, 1)
    x_grid = x.view(batch, 1).expand(batch, grid)
    gamma = jump_amplitude_torch(x, actions, cfg)
    x_jump = (x_grid + gamma).reshape(-1, 1)
    y1_jump = (y1.view(batch, 1).expand(batch, grid) + 1.0).reshape(-1, 1)
    y2_jump = y2.view(batch, 1).expand(batch, grid).reshape(-1, 1)
    u_jump = model(t_grid, x_jump, y1_jump, y2_jump).reshape(batch, grid)
    u_current = model(t, x, y1, y2).view(batch, 1)
    return u_jump - u_current


def sample_domain(dgm_cfg: DGMConfig, env_cfg: ErlangHawkesConfig, batch_size: int, device: torch.device):
    t = torch.rand(batch_size, 1, device=device) * env_cfg.horizon
    x = dgm_cfg.x_min + torch.rand(batch_size, 1, device=device) * (dgm_cfg.x_max - dgm_cfg.x_min)
    y1 = dgm_cfg.y_min + torch.rand(batch_size, 1, device=device) * (dgm_cfg.y_max - dgm_cfg.y_min)
    y2 = dgm_cfg.y_min + torch.rand(batch_size, 1, device=device) * (dgm_cfg.y_max - dgm_cfg.y_min)
    t.requires_grad_(True)
    x.requires_grad_(True)
    y1.requires_grad_(True)
    y2.requires_grad_(True)
    return t, x, y1, y2


def hjb_residual(model: DGMErlangOracle, t: torch.Tensor, x: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor):
    cfg = model.env_cfg
    dcfg = model.dgm_cfg
    u = model(t, x, y1, y2)
    u_t, u_x, u_y1, u_y2 = torch.autograd.grad(u.sum(), (t, x, y1, y2), create_graph=True)
    (u_xx,) = torch.autograd.grad(u_x.sum(), x, create_graph=True)

    actions = torch.linspace(0.0, 1.0, dcfg.action_grid_size, device=t.device).view(1, -1)
    lam = intensity_torch(x, y2, actions, cfg)
    jump_delta = jump_value_increment(model, t, x, y1, y2, actions)
    drift = state_drift_torch(x, actions, cfg)
    sigma = state_volatility_torch(x, actions, cfg)
    action_objective = cfg.control_cost * actions * actions + drift * u_x + 0.5 * sigma * sigma * u_xx + lam * jump_delta
    min_action_term, action_idx = torch.min(action_objective, dim=1, keepdim=True)

    residual = (
        u_t
        - cfg.erlang_decay * y1 * u_y1
        + (cfg.erlang_decay * y1 - cfg.erlang_decay * y2) * u_y2
        - cfg.beta * u
        + cfg.state_cost * x * x
        + min_action_term
    )
    best_actions = actions[0, action_idx.reshape(-1)].reshape(-1, 1)
    return residual, best_actions, u


def train_dgm_oracle(
    env_cfg: ErlangHawkesConfig | None = None,
    dgm_cfg: DGMConfig | None = None,
) -> Tuple[DGMErlangOracle, List[dict]]:
    env_cfg = env_cfg or ErlangHawkesConfig()
    dgm_cfg = dgm_cfg or DGMConfig()
    if dgm_cfg.device == "cuda" and not torch.cuda.is_available():
        dgm_cfg = DGMConfig(**{**dgm_cfg.__dict__, "device": "cpu"})
    torch.manual_seed(dgm_cfg.seed)
    np.random.seed(dgm_cfg.seed)
    device = torch.device(dgm_cfg.device)
    model = DGMErlangOracle(env_cfg, dgm_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=dgm_cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, dgm_cfg.steps // 3), gamma=0.55)

    history: List[dict] = []
    for step in range(1, dgm_cfg.steps + 1):
        t, x, y1, y2 = sample_domain(dgm_cfg, env_cfg, dgm_cfg.batch_size, device)
        residual, best_actions, u = hjb_residual(model, t, x, y1, y2)
        residual_loss = torch.mean(residual * residual)
        positivity_loss = torch.mean(torch.relu(-u) ** 2)
        loss = dgm_cfg.residual_weight * residual_loss + dgm_cfg.positivity_weight * positivity_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), dgm_cfg.grad_clip)
        opt.step()
        scheduler.step()

        if step == 1 or step % dgm_cfg.print_every == 0 or step == dgm_cfg.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "residual_loss": float(residual_loss.detach().cpu()),
                "positivity_loss": float(positivity_loss.detach().cpu()),
                "mean_action": float(best_actions.detach().mean().cpu()),
                "lr": float(scheduler.get_last_lr()[0]),
                "device": str(device),
            }
            history.append(row)
            print(
                "dgm_step={step:04d} loss={loss:9.5f} residual={residual_loss:9.5f} "
                "mean_action={mean_action:5.3f} lr={lr:.2e} device={device}".format(**row)
            )
    model.eval()
    return model, history
