from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ct_ddpg_powerlaw import collect_episode
from powerlaw_env import PowerLawHawkesConfig, powerlaw_kernel, readout_multiplier


def _signed_lstsq(features: np.ndarray, target: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    lhs = features.T @ features + ridge * np.eye(features.shape[1])
    rhs = features.T @ target
    return np.linalg.solve(lhs, rhs).astype(np.float64)


def stability_envelope_masses(env_cfg: PowerLawHawkesConfig) -> dict[str, float]:
    """Finite-horizon true and preselected-mixture envelope masses."""

    u = np.linspace(0.0, env_cfg.horizon, 2400)
    basis = np.exp(-np.outer(u, np.asarray(env_cfg.filter_betas, dtype=np.float64)))
    approximation_kernel = basis @ env_cfg.weights
    true_mass = env_cfg.excitation * (
        1.0 - (env_cfg.kernel_a / (env_cfg.kernel_a + env_cfg.horizon)) ** env_cfg.kernel_b
    )
    approximation_mass = env_cfg.excitation * np.trapz(np.abs(approximation_kernel), u)
    return {
        "true_envelope_mass": float(true_mass),
        "approximation_envelope_mass": float(approximation_mass),
    }


def fit_ct_memory_weights(
    env_cfg: PowerLawHawkesConfig,
    policy: Any,
    episodes: int = 40,
    seed: int = 7300,
) -> tuple[np.ndarray, dict]:
    """Fit an ex post signed readout from CT filters to true memory.

    In the power-law experiment, the rollout simulator uses exact event-history
    memory. The fitted weights are only a diagnostic for how well the finite
    filter bank can represent that memory.
    """

    features = []
    targets = []
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        ep = collect_episode(env_cfg, policy, rng, exploration_std=0.0)
        states = ep.states[:-1]
        actions = ep.actions
        baseline = np.clip(
            env_cfg.mu0 + env_cfg.mu_x * states[:, 0] + env_cfg.mu_a * actions,
            env_cfg.baseline_min,
            env_cfg.baseline_max,
        )
        multiplier = readout_multiplier(actions, env_cfg)
        denom = env_cfg.excitation * multiplier
        mask = np.abs(denom) > 1e-8
        if np.any(mask):
            features.append(states[mask, 1:])
            targets.append(np.maximum(0.0, (ep.intensities[mask] - baseline[mask]) / denom[mask]))

    x = np.vstack(features)
    y = np.concatenate(targets)
    signed_weights = _signed_lstsq(x, y)
    weights = signed_weights
    fitted = x @ weights
    fitted_positive = np.maximum(fitted, 0.0)
    summary = {
        "weights": weights,
        "onpolicy_weights": weights,
        "memory_rmse": float(np.sqrt(np.mean((fitted_positive - y) ** 2))),
        "signed_memory_rmse": float(np.sqrt(np.mean((fitted - y) ** 2))),
        "memory_relative_l2": float(np.linalg.norm(fitted_positive - y) / max(np.linalg.norm(y), 1e-12)),
        "samples": int(y.shape[0]),
    }
    return weights, summary


def plot_kernel_approximation_diagnostic(
    env_cfg: PowerLawHawkesConfig,
    policy: Any,
    save_path: str | Path | None = None,
    episodes: int = 40,
    seed: int = 7300,
):
    weights, summary = fit_ct_memory_weights(env_cfg, policy, episodes=episodes, seed=seed)
    u = np.linspace(0.0, env_cfg.horizon, 400)
    true_kernel = powerlaw_kernel(u, env_cfg.kernel_a, env_cfg.kernel_b)
    basis = np.exp(-np.outer(u, np.asarray(env_cfg.filter_betas, dtype=np.float64)))
    signed_kernel = basis @ weights
    fitted_kernel = np.maximum(signed_kernel, 0.0)
    preselected_kernel = basis @ env_cfg.weights
    kernel_rmse = float(np.sqrt(np.mean((fitted_kernel - true_kernel) ** 2)))
    kernel_relative_l2 = float(
        np.linalg.norm(fitted_kernel - true_kernel) / max(np.linalg.norm(true_kernel), 1e-12)
    )
    preselected_relative_l2 = float(
        np.linalg.norm(preselected_kernel - true_kernel) / max(np.linalg.norm(true_kernel), 1e-12)
    )
    summary.update(
        {
            "kernel_rmse": kernel_rmse,
            "kernel_relative_l2": kernel_relative_l2,
            "preselected_relative_l2": preselected_relative_l2,
            "preselected_weights": env_cfg.weights,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(u, true_kernel, label="true power-law kernel", linewidth=2.2)
    axes[0].plot(u, preselected_kernel, ":", label="global signed preselected fit", linewidth=2.2)
    axes[0].plot(u, fitted_kernel, "--", label="positive part of signed on-policy fit", linewidth=2.2)
    axes[0].set_xlabel("lag u")
    axes[0].set_ylabel("base kernel value")
    axes[0].set_title("Kernel approximation from CT-DDPG filter state")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    beta_values = np.asarray(env_cfg.filter_betas, dtype=np.float64)
    x = np.arange(len(beta_values))
    width = 0.38
    axes[1].bar(x - width / 2, weights, width=width, label="signed on-policy fit", color="#2563eb")
    axes[1].bar(x + width / 2, env_cfg.weights, width=width, label="global signed preselected", color="#f97316")
    axes[1].axhline(0.0, color="#111827", linewidth=0.8)
    tick_stride = max(1, int(np.ceil(len(beta_values) / 8)))
    tick_indices = np.arange(0, len(beta_values), tick_stride)
    if tick_indices[-1] != len(beta_values) - 1:
        if len(beta_values) - 1 - tick_indices[-1] < tick_stride:
            tick_indices[-1] = len(beta_values) - 1
        else:
            tick_indices = np.r_[tick_indices, len(beta_values) - 1]
    axes[1].set_xticks(tick_indices, [f"{beta_values[i]:.2f}" for i in tick_indices])
    axes[1].set_xlabel("filter beta")
    axes[1].set_ylabel("weight")
    axes[1].set_title(
        f"fit rel L2={kernel_relative_l2:.3f}, preselected rel L2={preselected_relative_l2:.3f}"
    )
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=160)
    return fig, summary
