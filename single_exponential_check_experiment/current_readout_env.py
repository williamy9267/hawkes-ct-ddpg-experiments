from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CurrentReadoutEnvConfig:
    """Single-exponential current-readout Hawkes jump-diffusion.

    State layout:
        state[0] = X_t
        state[1:] = online exponential filters Z_k(t)

    The default uses a small bank of filters. The true single-exponential
    Hawkes memory is the filter at `true_filter_index`; the remaining filters
    are extra model-free CT-DDPG state variables.
    """

    horizon: float = 5.0
    dt: float = 0.02
    beta: float = 0.02
    x0: float = 0.0
    z0: float = 0.0

    base_intensity: float = 2.05
    state_intensity_sensitivity: float = 0.10
    action_intensity_sensitivity: float = -0.08
    excitation: float = 1.25
    control_effect_strength: float = 1.25
    action_half_saturation: float = 0.35
    decay: float = 1.30
    # Arithmetic grid beta, 2 beta, ..., K beta; beta=0.325 makes the
    # fourth filter exactly the true single-exponential decay 1.30.
    filter_decays: Tuple[float, ...] = (0.325, 0.650, 0.975, 1.300, 1.625, 1.950, 2.275, 2.600)
    true_filter_index: int = 3
    baseline_min: float = 1e-6
    baseline_max: float = 6.0

    jump_size: float = 0.10
    jump_size_x_strength: float = 0.015
    jump_size_action_strength: float = -0.015
    jump_size_min: float = 1e-4
    jump_size_max: float = 0.25
    drift_reversion: float = 0.45
    drift_constant: float = 0.02
    diffusion: float = 0.05
    volatility_x_strength: float = 0.015
    control_drift_strength: float = 0.72
    control_volatility_strength: float = 0.08

    state_cost: float = 0.80
    control_cost: float = 0.18
    terminal_x_cost: float = 0.60

    x_scale: float = 2.0
    z_scale: float = 8.0

    @property
    def steps(self) -> int:
        return int(round(self.horizon / self.dt))

    @property
    def state_dim(self) -> int:
        return 1 + len(self.filter_decays)

    @property
    def feature_dim(self) -> int:
        return 1 + self.state_dim


def control_effectiveness(action: np.ndarray | float, cfg: CurrentReadoutEnvConfig) -> np.ndarray | float:
    action_arr = np.asarray(action, dtype=np.float64)
    return action_arr / (cfg.action_half_saturation + action_arr + 1e-12)


def readout_multiplier(action: np.ndarray | float, cfg: CurrentReadoutEnvConfig) -> np.ndarray | float:
    eff = control_effectiveness(action, cfg)
    return 1.0 - cfg.control_effect_strength * eff


def baseline_readout(x: float, action: float, cfg: CurrentReadoutEnvConfig) -> float:
    affine = cfg.base_intensity + cfg.state_intensity_sensitivity * x + cfg.action_intensity_sensitivity * action
    return float(np.clip(affine, cfg.baseline_min, cfg.baseline_max))


def current_readout_intensity(x: float, z: float, action: float, cfg: CurrentReadoutEnvConfig) -> float:
    baseline = baseline_readout(x, action, cfg)
    memory = cfg.excitation * readout_multiplier(action, cfg) * z
    return float(baseline + memory)


def state_drift(x: float, action: float, cfg: CurrentReadoutEnvConfig) -> float:
    return cfg.drift_constant - cfg.drift_reversion * x - cfg.control_drift_strength * action


def state_volatility(x: float, action: float, cfg: CurrentReadoutEnvConfig) -> float:
    return float(max(1e-6, cfg.diffusion + cfg.volatility_x_strength * x + cfg.control_volatility_strength * action))


def jump_amplitude(x: float, action: float, cfg: CurrentReadoutEnvConfig) -> float:
    raw = cfg.jump_size + cfg.jump_size_x_strength * x + cfg.jump_size_action_strength * action
    return float(np.clip(raw, cfg.jump_size_min, cfg.jump_size_max))


class SingleExpCurrentReadoutEnv:
    """Euler simulator for Section 1's current-action readout model."""

    def __init__(self, cfg: CurrentReadoutEnvConfig):
        self.cfg = cfg
        self.filter_decays = np.asarray(cfg.filter_decays, dtype=np.float64)
        self.reset()

    def reset(self) -> np.ndarray:
        self.k = 0
        self.t = 0.0
        self.x = float(self.cfg.x0)
        self.z_filters = np.full(len(self.cfg.filter_decays), self.cfg.z0, dtype=np.float64)
        return self.observation()

    def set_state(self, time: float, state: np.ndarray) -> np.ndarray:
        state_arr = np.asarray(state, dtype=np.float64).reshape(-1)
        if state_arr.shape[0] != self.cfg.state_dim:
            raise ValueError(f"state must have length {self.cfg.state_dim}, got {state_arr.shape[0]}")
        self.k = int(np.clip(round(time / self.cfg.dt), 0, self.cfg.steps))
        self.t = min(self.cfg.horizon, self.k * self.cfg.dt)
        self.x = float(state_arr[0])
        self.z_filters = state_arr[1:].copy()
        return self.observation()

    def observation(self) -> np.ndarray:
        return np.concatenate([[self.x], self.z_filters.copy()])

    def true_memory(self) -> float:
        return float(self.z_filters[self.cfg.true_filter_index])

    def intensity(self, action: float) -> float:
        return current_readout_intensity(self.x, self.true_memory(), action, self.cfg)

    def terminal_reward(self, states: np.ndarray) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float64)
        return -self.cfg.terminal_x_cost * states_arr[:, 0] ** 2

    def step(self, action: float, rng: np.random.Generator):
        cfg = self.cfg
        action = float(np.clip(action, 0.0, 1.0))
        lam = self.intensity(action)
        jump_count = int(rng.poisson(lam * cfg.dt))

        x_prev = self.x
        drift = state_drift(x_prev, action, cfg)
        sigma = state_volatility(x_prev, action, cfg)
        gamma = jump_amplitude(x_prev, action, cfg)
        diffusion_step = sigma * np.sqrt(cfg.dt) * rng.normal()
        x_next = x_prev + drift * cfg.dt + diffusion_step + gamma * jump_count

        z_next = self.z_filters * np.exp(-self.filter_decays * cfg.dt)
        if jump_count > 0:
            event_offsets = rng.uniform(0.0, cfg.dt, size=jump_count)
            ages_to_grid_end = cfg.dt - event_offsets
            z_next += np.exp(-np.outer(self.filter_decays, ages_to_grid_end)).sum(axis=1)

        running_cost = cfg.state_cost * x_prev * x_prev + cfg.control_cost * action * action
        reward_rate = -running_cost

        self.x = float(x_next)
        self.z_filters = z_next.astype(np.float64)
        self.k += 1
        self.t = min(cfg.horizon, self.k * cfg.dt)
        done = self.k >= cfg.steps
        info = {
            "lambda": lam,
            "jumps": jump_count,
            "z": self.true_memory(),
            "z_norm": float(np.linalg.norm(self.z_filters)),
            "gamma": gamma,
            "cost_rate": -reward_rate,
        }
        return self.observation(), float(reward_rate), done, info


def state_features(times: np.ndarray, states: np.ndarray, cfg: CurrentReadoutEnvConfig) -> np.ndarray:
    times_arr = np.asarray(times, dtype=np.float64)
    states_arr = np.asarray(states, dtype=np.float64)
    t_feat = 2.0 * times_arr[..., None] / cfg.horizon - 1.0
    x_feat = np.clip(states_arr[..., 0:1] / cfg.x_scale, -5.0, 5.0)
    z_feat = np.clip(states_arr[..., 1:] / cfg.z_scale, -5.0, 5.0)
    return np.concatenate([t_feat, x_feat, z_feat], axis=-1)
