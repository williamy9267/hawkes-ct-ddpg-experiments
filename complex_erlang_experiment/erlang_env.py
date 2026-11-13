from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ErlangHawkesConfig:
    horizon: float = 5.0
    dt: float = 0.02
    beta: float = 0.02
    x0: float = 0.0
    y10: float = 0.0
    y20: float = 0.0
    z0: float = 0.0

    # Current-readout Hawkes intensity with an Erlang-2 true kernel.
    mu0: float = 2.00
    mu_x: float = 0.03
    mu_a: float = -0.04
    excitation: float = 1.05
    control_effect_strength: float = 1.30
    action_half_saturation: float = 0.35
    erlang_decay: float = 1.15
    baseline_min: float = 1e-6
    baseline_max: float = 5.0

    # Model-free exponential filters used by CT-DDPG and optional fair baselines.
    # Arithmetic grid beta, 2 beta, ..., K beta. Here beta=0.23 so that the
    # fifth filter is centered at the Erlang decay rho=1.15.
    filter_decays: Tuple[float, ...] = (
        0.23,
        0.46,
        0.69,
        0.92,
        1.15,
        1.38,
        1.61,
        1.84,
        2.07,
        2.30,
        2.53,
        2.76,
    )

    # Controlled jump-diffusion state.
    drift_constant: float = 0.02
    drift_reversion: float = 0.45
    control_drift_strength: float = 0.70
    diffusion: float = 0.05
    volatility_x_strength: float = 0.012
    control_volatility_strength: float = 0.08
    jump_size: float = 0.085
    jump_size_x_strength: float = 0.005
    jump_size_action_strength: float = -0.010
    jump_size_min: float = 1e-4
    jump_size_max: float = 0.20
    # Bernoulli two-point mark: +1 with this probability and -1 otherwise.
    # The default 1.0 recovers the original unmarked positive-jump experiment.
    jump_mark_positive_probability: float = 1.0

    # Common 0.25 normalization preserves the optimizer and memory gap while
    # improving critic and PINN conditioning.
    state_cost: float = 0.3000
    control_cost: float = 0.2250
    terminal_x_cost: float = 0.2250

    x_scale: float = 2.0
    y_scale: float = 8.0
    z_scale: float = 10.0

    @property
    def steps(self) -> int:
        return int(round(self.horizon / self.dt))

    @property
    def state_dim(self) -> int:
        # X, exact Erlang phases Y1/Y2, and model-free exponential filters Z^k.
        return 3 + len(self.filter_decays)

    @property
    def method_feature_dim(self) -> int:
        # time, X, and exponential filters only. Exact Erlang phases are hidden.
        return 2 + len(self.filter_decays)


def control_effectiveness(action: np.ndarray | float, cfg: ErlangHawkesConfig) -> np.ndarray | float:
    action_arr = np.asarray(action, dtype=np.float64)
    return action_arr / (cfg.action_half_saturation + action_arr + 1e-12)


def readout_multiplier(action: np.ndarray | float, cfg: ErlangHawkesConfig) -> np.ndarray | float:
    return 1.0 - cfg.control_effect_strength * control_effectiveness(action, cfg)


def state_drift(x: float, action: float, cfg: ErlangHawkesConfig) -> float:
    return cfg.drift_constant - cfg.drift_reversion * x - cfg.control_drift_strength * action


def state_volatility(x: float, action: float, cfg: ErlangHawkesConfig) -> float:
    return float(max(1e-6, cfg.diffusion + cfg.volatility_x_strength * x + cfg.control_volatility_strength * action))


def jump_amplitude(x: float, action: float, cfg: ErlangHawkesConfig) -> float:
    raw = cfg.jump_size + cfg.jump_size_x_strength * x + cfg.jump_size_action_strength * action
    return float(np.clip(raw, cfg.jump_size_min, cfg.jump_size_max))


def erlang_intensity(x: float, y2: float, action: float, cfg: ErlangHawkesConfig) -> float:
    affine_baseline = cfg.mu0 + cfg.mu_x * x + cfg.mu_a * action
    baseline = np.clip(affine_baseline, cfg.baseline_min, cfg.baseline_max)
    memory = cfg.excitation * readout_multiplier(action, cfg) * y2
    return float(baseline + memory)


class ErlangHawkesEnv:
    """Single-dimensional controlled Hawkes jump-diffusion with Erlang-2 memory.

    True hidden Hawkes memory:
        dY1 = -rho Y1 dt + dN_t
        dY2 =  rho Y1 dt - rho Y2 dt

    The intensity uses Y2. CT-DDPG does not observe Y1/Y2 directly; it observes
    X and an exponential filter bank Z updated from event times.
    """

    def __init__(self, cfg: ErlangHawkesConfig):
        self.cfg = cfg
        self.filter_decays = np.asarray(cfg.filter_decays, dtype=np.float64)
        self.reset()

    def reset(self) -> np.ndarray:
        self.k = 0
        self.t = 0.0
        self.x = float(self.cfg.x0)
        self.y1 = float(self.cfg.y10)
        self.y2 = float(self.cfg.y20)
        self.z = np.full(len(self.cfg.filter_decays), self.cfg.z0, dtype=np.float64)
        return self.observation()

    def set_state(self, time: float, state: np.ndarray) -> np.ndarray:
        state_arr = np.asarray(state, dtype=np.float64).reshape(-1)
        if state_arr.shape[0] != self.cfg.state_dim:
            raise ValueError(f"state must have length {self.cfg.state_dim}, got {state_arr.shape[0]}")
        self.k = int(np.clip(round(time / self.cfg.dt), 0, self.cfg.steps))
        self.t = min(self.cfg.horizon, self.k * self.cfg.dt)
        self.x = float(state_arr[0])
        self.y1 = float(state_arr[1])
        self.y2 = float(state_arr[2])
        self.z = state_arr[3:].copy()
        return self.observation()

    def observation(self) -> np.ndarray:
        return np.concatenate([[self.x, self.y1, self.y2], self.z.copy()])

    def intensity(self, action: float) -> float:
        return erlang_intensity(self.x, self.y2, action, self.cfg)

    def terminal_reward(self, states: np.ndarray) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float64)
        return -self.cfg.terminal_x_cost * states_arr[:, 0] ** 2

    def step(self, action: float, rng: np.random.Generator):
        cfg = self.cfg
        action = float(np.clip(action, 0.0, 1.0))
        lam = self.intensity(action)
        jump_count = int(rng.poisson(lam * cfg.dt))

        x_prev = self.x
        y1_prev = self.y1
        y2_prev = self.y2

        drift = state_drift(x_prev, action, cfg)
        sigma = state_volatility(x_prev, action, cfg)
        gamma = jump_amplitude(x_prev, action, cfg)
        diffusion_step = sigma * np.sqrt(cfg.dt) * rng.normal()
        positive_marks = int(rng.binomial(jump_count, cfg.jump_mark_positive_probability))
        jump_mark_sum = 2 * positive_marks - jump_count
        x_next = x_prev + drift * cfg.dt + diffusion_step + gamma * jump_mark_sum

        rho = cfg.erlang_decay
        exp_rdt = np.exp(-rho * cfg.dt)
        y1_next = exp_rdt * y1_prev
        y2_next = exp_rdt * (y2_prev + rho * cfg.dt * y1_prev)
        z_next = self.z * np.exp(-self.filter_decays * cfg.dt)

        if jump_count > 0:
            event_offsets = rng.uniform(0.0, cfg.dt, size=jump_count)
            ages = cfg.dt - event_offsets
            y1_next += np.exp(-rho * ages).sum()
            y2_next += (rho * ages * np.exp(-rho * ages)).sum()
            z_next += np.exp(-np.outer(self.filter_decays, ages)).sum(axis=1)

        running_cost = cfg.state_cost * x_prev * x_prev + cfg.control_cost * action * action
        reward_rate = -running_cost

        self.x = float(x_next)
        self.y1 = float(y1_next)
        self.y2 = float(y2_next)
        self.z = z_next.astype(np.float64)
        self.k += 1
        self.t = min(cfg.horizon, self.k * cfg.dt)
        done = self.k >= cfg.steps
        info = {
            "lambda": lam,
            "jumps": jump_count,
            "jump_mark_sum": jump_mark_sum,
            "y1": self.y1,
            "y2": self.y2,
            "z_norm": float(np.linalg.norm(self.z)),
            "gamma": gamma,
            "cost_rate": -reward_rate,
        }
        return self.observation(), float(reward_rate), done, info


def lifted_filter_features(filters: np.ndarray, cfg: ErlangHawkesConfig) -> np.ndarray:
    """Condition one observed filter bank without adding state information."""

    filters_arr = np.maximum(np.asarray(filters, dtype=np.float64), 0.0)
    decays = np.asarray(cfg.filter_decays, dtype=np.float64)
    if filters_arr.shape[-1] != decays.size:
        raise ValueError("the last filter dimension must match filter_decays")
    levels = np.log1p(decays * filters_arr)
    if decays.size <= 1:
        return levels
    gaps = np.diff(decays)
    if np.any(gaps <= 0.0):
        raise ValueError("filter_decays must be strictly increasing for lifted features")
    midpoints = 0.5 * (decays[:-1] + decays[1:])
    age_memory = midpoints * np.maximum(filters_arr[..., :-1] - filters_arr[..., 1:], 0.0) / gaps
    return np.concatenate([levels, np.log1p(age_memory)], axis=-1)


def method_state_features(
    times: np.ndarray,
    states: np.ndarray,
    cfg: ErlangHawkesConfig,
    feature_mode: str = "matched",
) -> np.ndarray:
    times_arr = np.asarray(times, dtype=np.float64)
    states_arr = np.asarray(states, dtype=np.float64)
    t_feat = 2.0 * times_arr[..., None] / cfg.horizon - 1.0
    x_feat = np.clip(states_arr[..., 0:1] / cfg.x_scale, -5.0, 5.0)
    filters = states_arr[..., 3 : 3 + len(cfg.filter_decays)]
    if feature_mode == "matched":
        z_feat = np.clip(filters / cfg.z_scale, -5.0, 5.0)
    elif feature_mode == "lifted":
        z_feat = np.clip(lifted_filter_features(filters, cfg), 0.0, 5.0)
    else:
        raise ValueError("feature_mode must be 'matched' or 'lifted'")
    return np.concatenate([t_feat, x_feat, z_feat], axis=-1)


def exact_oracle_state_features(times: np.ndarray, states: np.ndarray, cfg: ErlangHawkesConfig) -> np.ndarray:
    times_arr = np.asarray(times, dtype=np.float64)
    states_arr = np.asarray(states, dtype=np.float64)
    t_feat = 2.0 * times_arr[..., None] / cfg.horizon - 1.0
    x_feat = np.clip(states_arr[..., 0:1] / cfg.x_scale, -5.0, 5.0)
    y_feat = np.clip(states_arr[..., 1:3] / cfg.y_scale, -5.0, 5.0)
    return np.concatenate([t_feat, x_feat, y_feat], axis=-1)
