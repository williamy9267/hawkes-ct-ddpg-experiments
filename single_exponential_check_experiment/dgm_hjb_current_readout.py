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

from current_readout_env import CurrentReadoutEnvConfig


@dataclass
class DGMConfig:
    hidden_dim: int = 64
    num_layers: int = 3
    lr: float = 2e-3
    steps: int = 450
    batch_size: int = 256
    action_grid_size: int = 41
    x_min: float = -1.2
    x_max: float = 3.2
    z_min: float = 0.0
    z_max: float = 8.0
    action_fd_eps: float = 0.02
    residual_weight: float = 1.0
    positivity_weight: float = 1e-3
    grad_clip: float = 5.0
    seed: int = 123
    device: str = "cpu"
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


def control_effectiveness_torch(actions: torch.Tensor, cfg: CurrentReadoutEnvConfig) -> torch.Tensor:
    return actions / (cfg.action_half_saturation + actions + 1e-12)


def readout_multiplier_torch(actions: torch.Tensor, cfg: CurrentReadoutEnvConfig) -> torch.Tensor:
    return 1.0 - cfg.control_effect_strength * control_effectiveness_torch(actions, cfg)


def intensity_torch(x: torch.Tensor, z: torch.Tensor, actions: torch.Tensor, cfg: CurrentReadoutEnvConfig) -> torch.Tensor:
    affine_baseline = cfg.base_intensity + cfg.state_intensity_sensitivity * x + cfg.action_intensity_sensitivity * actions
    baseline = torch.clamp(affine_baseline, min=cfg.baseline_min, max=cfg.baseline_max)
    memory = cfg.excitation * readout_multiplier_torch(actions, cfg) * z
    return baseline + memory


def state_drift_torch(x: torch.Tensor, actions: torch.Tensor, cfg: CurrentReadoutEnvConfig) -> torch.Tensor:
    return cfg.drift_constant - cfg.drift_reversion * x - cfg.control_drift_strength * actions


def state_volatility_torch(x: torch.Tensor, actions: torch.Tensor, cfg: CurrentReadoutEnvConfig) -> torch.Tensor:
    return torch.clamp(
        cfg.diffusion + cfg.volatility_x_strength * x + cfg.control_volatility_strength * actions,
        min=1e-6,
    )


def jump_amplitude_torch(x: torch.Tensor, actions: torch.Tensor, cfg: CurrentReadoutEnvConfig) -> torch.Tensor:
    raw = cfg.jump_size + cfg.jump_size_x_strength * x + cfg.jump_size_action_strength * actions
    return torch.clamp(raw, min=cfg.jump_size_min, max=cfg.jump_size_max)


def exact_z_from_states(states: np.ndarray, cfg: CurrentReadoutEnvConfig) -> np.ndarray:
    states_arr = np.asarray(states, dtype=np.float64)
    if states_arr.shape[1] == 2:
        return states_arr[:, 1]
    return states_arr[:, 1 + cfg.true_filter_index]


class DGMHJBCurrentReadout(nn.Module):
    """Known-parameter HJB oracle for the exact single-exponential state (X, Z)."""

    def __init__(self, env_cfg: CurrentReadoutEnvConfig, dgm_cfg: DGMConfig):
        super().__init__()
        self.env_cfg = env_cfg
        self.dgm_cfg = dgm_cfg
        self.net = DGMNet(3, dgm_cfg.hidden_dim, dgm_cfg.num_layers)

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
            z = torch.as_tensor(exact_z_from_states(states, self.env_cfg), dtype=torch.float32, device=device).reshape(-1, 1)
            return self(t, x, z).cpu().numpy().reshape(-1)

    def action_np(self, times: np.ndarray, states: np.ndarray) -> np.ndarray:
        self.eval()
        cfg = self.env_cfg
        dcfg = self.dgm_cfg
        with torch.no_grad():
            device = next(self.parameters()).device
            t = torch.as_tensor(times, dtype=torch.float32, device=device).reshape(-1, 1)
            x = torch.as_tensor(states[:, 0], dtype=torch.float32, device=device).reshape(-1, 1)
            z = torch.as_tensor(exact_z_from_states(states, self.env_cfg), dtype=torch.float32, device=device).reshape(-1, 1)
            eps = dcfg.action_fd_eps
            u = self(t, x, z)
            u_plus = self(t, x + eps, z)
            u_minus = self(t, x - eps, z)
            u_x = (u_plus - u_minus) / (2.0 * eps)
            u_xx = (u_plus - 2.0 * u + u_minus) / (eps * eps)

            actions = torch.linspace(0.0, 1.0, dcfg.action_grid_size, device=device).view(1, -1)
            lam = intensity_torch(x, z, actions, cfg)
            jump_delta = jump_value_increment(model=self, t=t, x=x, z=z, actions=actions)
            drift = state_drift_torch(x, actions, cfg)
            sigma = state_volatility_torch(x, actions, cfg)
            objective = (
                cfg.control_cost * actions * actions
                + drift * u_x
                + 0.5 * sigma * sigma * u_xx
                + lam * jump_delta
            )
            best = torch.argmin(objective, dim=1)
            return actions[0, best].cpu().numpy()


class DGMOraclePolicy:
    def __init__(self, oracle: DGMHJBCurrentReadout):
        self.oracle = oracle

    def act(self, time: float, state: np.ndarray) -> float:
        actions = self.oracle.action_np(np.asarray([time]), np.asarray([state], dtype=np.float64))
        return float(actions[0])

    def value_path(self, times: np.ndarray, states: np.ndarray) -> np.ndarray:
        return self.oracle.value_np(times, states)


def jump_value_increment(
    model: DGMHJBCurrentReadout,
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
    z_jump = (z.view(batch, 1).expand(batch, grid) + 1.0).reshape(-1, 1)
    u_jump = model(t_grid, x_jump, z_jump).reshape(batch, grid)
    u_current = model(t, x, z).view(batch, 1)
    return u_jump - u_current


def sample_domain(dgm_cfg: DGMConfig, env_cfg: CurrentReadoutEnvConfig, batch_size: int, device: torch.device):
    t = torch.rand(batch_size, 1, device=device) * env_cfg.horizon
    x = dgm_cfg.x_min + torch.rand(batch_size, 1, device=device) * (dgm_cfg.x_max - dgm_cfg.x_min)
    z = dgm_cfg.z_min + torch.rand(batch_size, 1, device=device) * (dgm_cfg.z_max - dgm_cfg.z_min)
    t.requires_grad_(True)
    x.requires_grad_(True)
    z.requires_grad_(True)
    return t, x, z


def hjb_residual(model: DGMHJBCurrentReadout, t: torch.Tensor, x: torch.Tensor, z: torch.Tensor):
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
    action_objective = (
        cfg.control_cost * actions * actions
        + drift * u_x
        + 0.5 * sigma * sigma * u_xx
        + lam * jump_delta
    )
    min_action_term, action_idx = torch.min(action_objective, dim=1, keepdim=True)

    residual = (
        u_t
        - cfg.decay * z * u_z
        - cfg.beta * u
        + cfg.state_cost * x * x
        + min_action_term
    )
    best_actions = actions[0, action_idx.reshape(-1)].reshape(-1, 1)
    return residual, best_actions, u


def train_dgm_oracle(
    env_cfg: CurrentReadoutEnvConfig | None = None,
    dgm_cfg: DGMConfig | None = None,
) -> Tuple[DGMHJBCurrentReadout, List[dict]]:
    env_cfg = env_cfg or CurrentReadoutEnvConfig()
    dgm_cfg = dgm_cfg or DGMConfig()
    torch.manual_seed(dgm_cfg.seed)
    np.random.seed(dgm_cfg.seed)
    device = torch.device(dgm_cfg.device)
    model = DGMHJBCurrentReadout(env_cfg, dgm_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=dgm_cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, dgm_cfg.steps // 3), gamma=0.55)

    history: List[dict] = []
    for step in range(1, dgm_cfg.steps + 1):
        t, x, z = sample_domain(dgm_cfg, env_cfg, dgm_cfg.batch_size, device)
        residual, best_actions, u = hjb_residual(model, t, x, z)
        residual_loss = torch.mean(residual * residual)
        positivity_loss = torch.mean(torch.relu(-u) ** 2)
        loss = dgm_cfg.residual_weight * residual_loss + dgm_cfg.positivity_weight * positivity_loss

        opt.zero_grad()
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
            }
            history.append(row)
            print(
                "dgm_step={step:04d} loss={loss:9.5f} residual={residual_loss:9.5f} "
                "mean_action={mean_action:5.3f} lr={lr:.2e}".format(**row)
            )
    model.eval()
    return model, history
