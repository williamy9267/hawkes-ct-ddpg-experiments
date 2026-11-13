from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import numpy as np


def positive_part(x: np.ndarray | float) -> np.ndarray | float:
    return np.maximum(x, 0.0)


def powerlaw_kernel(u: np.ndarray | float, a: float, b: float) -> np.ndarray | float:
    u_arr = np.asarray(u, dtype=np.float64)
    return b * (a**b) * np.power(u_arr + a, -(1.0 + b))


def fit_powerlaw_mixture_weights(
    betas: Tuple[float, ...],
    kernel_a: float,
    kernel_b: float,
    horizon: float,
    dt: float,
) -> Tuple[float, ...]:
    """Signed ridge fit phi(u) ~= sum_k w_k exp(-beta_k u).

    The approximated intensity applies a positive part after the memory
    aggregation, so signed finite-dictionary coefficients are admissible.
    """

    beta_arr = np.asarray(betas, dtype=np.float64)
    u_grid = np.linspace(0.0, horizon, max(600, int(round(horizon / dt)) * 4))
    design = np.exp(-np.outer(u_grid, beta_arr))
    target = powerlaw_kernel(u_grid, kernel_a, kernel_b)
    ridge = 1e-4
    weights = np.linalg.solve(design.T @ design + ridge * np.eye(len(beta_arr)), design.T @ target)
    return tuple(float(x) for x in weights)


@dataclass
class PowerLawHawkesConfig:
    horizon: float = 8.0
    dt: float = 0.02
    beta: float = 0.02
    x0: float = 0.0
    z0: float = 0.0

    # Power-law kernel phi(u)=b*a^b*(u+a)^(-(1+b)) approximated by exponentials.
    kernel_a: float = 0.12
    kernel_b: float = 0.80
    # Arithmetic grid beta, 2 beta, ..., K beta to match the density-style
    # exponential dictionary construction. The sharp kernel needs beta=1.
    filter_betas: Tuple[float, ...] = (
        1.00,
        2.00,
        3.00,
        4.00,
        5.00,
        6.00,
        7.00,
        8.00,
        9.00,
        10.00,
        11.00,
        12.00,
        13.00,
        14.00,
        15.00,
        16.00,
        17.00,
        18.00,
        19.00,
        20.00,
    )
    mixture_weights: Tuple[float, ...] | None = None

    # Controlled current-readout Hawkes intensity.
    mu0: float = 0.60
    mu_x: float = 0.002
    mu_a: float = -0.002
    excitation: float = 0.99
    control_effect_strength: float = 1.08
    action_half_saturation: float = 0.10
    baseline_min: float = 1e-6
    baseline_max: float = 5.0

    # Controlled jump-diffusion state.
    drift_constant: float = 0.0
    drift_reversion: float = 1.00
    control_drift_strength: float = 0.02
    diffusion: float = 0.03
    volatility_x_strength: float = 0.002
    control_volatility_strength: float = 0.005
    jump_size: float = 0.50
    jump_size_x_strength: float = 0.002
    jump_size_action_strength: float = -0.05
    jump_size_min: float = 1e-4
    jump_size_max: float = 0.90
    # Bernoulli two-point mark: +1 with this probability and -1 otherwise.
    # The default 1.0 recovers the original unmarked positive-jump experiment.
    jump_mark_positive_probability: float = 1.0

    # Sharp clustered-event calibration: constant suppression is costly, while
    # filter-triggered intervention is valuable during endogenous bursts.
    state_cost: float = 1.00
    control_cost: float = 0.50
    terminal_x_cost: float = 0.50

    x_scale: float = 1.0
    z_scale: float = 1.0

    @property
    def steps(self) -> int:
        return int(round(self.horizon / self.dt))

    @property
    def state_dim(self) -> int:
        # X plus K exponential-filter states.
        return 1 + len(self.filter_betas)

    @property
    def method_feature_dim(self) -> int:
        # time, X, and the same K filters.
        return 2 + len(self.filter_betas)

    @cached_property
    def weights(self) -> np.ndarray:
        if self.mixture_weights is not None:
            weights = np.asarray(self.mixture_weights, dtype=np.float64)
        else:
            weights = np.asarray(
                fit_powerlaw_mixture_weights(
                    self.filter_betas,
                    self.kernel_a,
                    self.kernel_b,
                    self.horizon,
                    self.dt,
                ),
                dtype=np.float64,
            )
        if weights.shape[0] != len(self.filter_betas):
            raise ValueError("mixture_weights must have the same length as filter_betas")
        return weights


def control_effectiveness(action: np.ndarray | float, cfg: PowerLawHawkesConfig) -> np.ndarray | float:
    action_arr = np.asarray(action, dtype=np.float64)
    return action_arr / (cfg.action_half_saturation + action_arr + 1e-12)


def readout_multiplier(action: np.ndarray | float, cfg: PowerLawHawkesConfig) -> np.ndarray | float:
    return 1.0 - cfg.control_effect_strength * control_effectiveness(action, cfg)


def state_drift(x: float, action: float, cfg: PowerLawHawkesConfig) -> float:
    return cfg.drift_constant - cfg.drift_reversion * x - cfg.control_drift_strength * action


def state_volatility(x: float, action: float, cfg: PowerLawHawkesConfig) -> float:
    return float(max(1e-6, cfg.diffusion + cfg.volatility_x_strength * x + cfg.control_volatility_strength * action))


def jump_amplitude(x: float, action: float, cfg: PowerLawHawkesConfig) -> float:
    raw = cfg.jump_size + cfg.jump_size_x_strength * x + cfg.jump_size_action_strength * action
    return float(np.clip(raw, cfg.jump_size_min, cfg.jump_size_max))


def kernel_readout(z: np.ndarray, cfg: PowerLawHawkesConfig) -> float:
    return float(np.dot(cfg.weights, np.asarray(z, dtype=np.float64)))


def baseline_readout(x: float, action: float, cfg: PowerLawHawkesConfig) -> float:
    affine_baseline = cfg.mu0 + cfg.mu_x * x + cfg.mu_a * action
    return float(np.clip(affine_baseline, cfg.baseline_min, cfg.baseline_max))


def classical_powerlaw_intensity(x: float, memory: float, action: float, cfg: PowerLawHawkesConfig) -> float:
    baseline = baseline_readout(x, action, cfg)
    memory_term = cfg.excitation * readout_multiplier(action, cfg) * memory
    return float(baseline + memory_term)


def approximate_powerlaw_intensity(x: float, z: np.ndarray, action: float, cfg: PowerLawHawkesConfig) -> float:
    baseline = baseline_readout(x, action, cfg)
    memory = cfg.excitation * readout_multiplier(action, cfg) * kernel_readout(z, cfg)
    return float(baseline + positive_part(memory))


class PowerLawHawkesEnv:
    """Controlled jump-diffusion with exact history-based power-law Hawkes memory.

    The true simulation intensity uses the classical positive kernel
        phi(u)=b*a^b*(u+a)^(-(1+b))
    applied to the full observed event history. The finite filters
        dZ^k_t = -beta_k Z^k_t dt + dN_t,
    are approximate Markov features for CT-DDPG and the DGM/HJB approximation;
    they are not the true power-law memory.
    """

    def __init__(self, cfg: PowerLawHawkesConfig):
        self.cfg = cfg
        self.filter_betas = np.asarray(cfg.filter_betas, dtype=np.float64)
        self.reset()

    def reset(self) -> np.ndarray:
        self.k = 0
        self.t = 0.0
        self.x = float(self.cfg.x0)
        self.z = np.full(len(self.cfg.filter_betas), self.cfg.z0, dtype=np.float64)
        self.event_times: list[float] = []
        self.history_available = True
        return self.observation()

    def set_state(self, time: float, state: np.ndarray) -> np.ndarray:
        state_arr = np.asarray(state, dtype=np.float64).reshape(-1)
        if state_arr.shape[0] != self.cfg.state_dim:
            raise ValueError(f"state must have length {self.cfg.state_dim}, got {state_arr.shape[0]}")
        self.k = int(np.clip(round(time / self.cfg.dt), 0, self.cfg.steps))
        self.t = min(self.cfg.horizon, self.k * self.cfg.dt)
        self.x = float(state_arr[0])
        self.z = state_arr[1:].copy()
        self.event_times = []
        # A generic Markov state does not identify a unique power-law history.
        # Rollouts from set_state therefore use the finite approximation.
        self.history_available = False
        return self.observation()

    def observation(self) -> np.ndarray:
        return np.concatenate([[self.x], self.z.copy()])

    def true_kernel_readout(self) -> float:
        if not self.event_times:
            return 0.0
        event_times = np.asarray(self.event_times, dtype=np.float64)
        ages = np.maximum(self.t - event_times, 0.0)
        return float(np.sum(powerlaw_kernel(ages, self.cfg.kernel_a, self.cfg.kernel_b)))

    def approximate_kernel_readout(self) -> float:
        return kernel_readout(self.z, self.cfg)

    def intensity(self, action: float) -> float:
        if self.history_available:
            return classical_powerlaw_intensity(self.x, self.true_kernel_readout(), action, self.cfg)
        return approximate_powerlaw_intensity(self.x, self.z, action, self.cfg)

    def terminal_reward(self, states: np.ndarray) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float64)
        return -self.cfg.terminal_x_cost * states_arr[:, 0] ** 2

    def step(self, action: float, rng: np.random.Generator):
        cfg = self.cfg
        action = float(np.clip(action, 0.0, 1.0))
        lam = self.intensity(action)
        true_readout_before = self.true_kernel_readout() if self.history_available else np.nan
        approx_readout_before = self.approximate_kernel_readout()
        jump_count = int(rng.poisson(lam * cfg.dt))

        x_prev = self.x
        drift = state_drift(x_prev, action, cfg)
        sigma = state_volatility(x_prev, action, cfg)
        gamma = jump_amplitude(x_prev, action, cfg)
        diffusion_step = sigma * np.sqrt(cfg.dt) * rng.normal()
        positive_marks = int(rng.binomial(jump_count, cfg.jump_mark_positive_probability))
        jump_mark_sum = 2 * positive_marks - jump_count
        x_next = x_prev + drift * cfg.dt + diffusion_step + gamma * jump_mark_sum

        z_next = self.z * np.exp(-self.filter_betas * cfg.dt)
        if jump_count > 0:
            event_offsets = rng.uniform(0.0, cfg.dt, size=jump_count)
            ages = cfg.dt - event_offsets
            z_next += np.exp(-np.outer(self.filter_betas, ages)).sum(axis=1)
            if self.history_available:
                self.event_times.extend(float(self.t + offset) for offset in event_offsets)

        running_cost = cfg.state_cost * x_prev * x_prev + cfg.control_cost * action * action
        reward_rate = -running_cost

        self.x = float(x_next)
        self.z = z_next.astype(np.float64)
        self.k += 1
        self.t = min(cfg.horizon, self.k * cfg.dt)
        done = self.k >= cfg.steps
        info = {
            "lambda": lam,
            "jumps": jump_count,
            "jump_mark_sum": jump_mark_sum,
            "kernel_readout": self.approximate_kernel_readout(),
            "approx_kernel_readout": self.approximate_kernel_readout(),
            "true_kernel_readout": self.true_kernel_readout() if self.history_available else np.nan,
            "true_kernel_readout_before": true_readout_before,
            "approx_kernel_readout_before": approx_readout_before,
            "z_norm": float(np.linalg.norm(self.z)),
            "gamma": gamma,
            "cost_rate": -reward_rate,
        }
        return self.observation(), float(reward_rate), done, info


# Backward-compatible alias for code that explicitly wants the finite approximation.
powerlaw_intensity = approximate_powerlaw_intensity


def method_state_features(times: np.ndarray, states: np.ndarray, cfg: PowerLawHawkesConfig) -> np.ndarray:
    times_arr = np.asarray(times, dtype=np.float64)
    states_arr = np.asarray(states, dtype=np.float64)
    t_feat = 2.0 * times_arr[..., None] / cfg.horizon - 1.0
    x_feat = np.clip(states_arr[..., 0:1] / cfg.x_scale, -5.0, 5.0)
    z_feat = np.clip(states_arr[..., 1:] / cfg.z_scale, -5.0, 5.0)
    return np.concatenate([t_feat, x_feat, z_feat], axis=-1)
