from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ct_ddpg_current_readout import collect_episode
from current_readout_env import CurrentReadoutEnvConfig, readout_multiplier


def _signed_lstsq(features: np.ndarray, target: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    lhs = features.T @ features + ridge * np.eye(features.shape[1])
    rhs = features.T @ target
    return np.linalg.solve(lhs, rhs).astype(np.float64)


def single_exp_kernel(u: np.ndarray, cfg: CurrentReadoutEnvConfig) -> np.ndarray:
    return np.exp(-cfg.decay * np.asarray(u, dtype=np.float64))


def global_kernel_fit_weights(env_cfg: CurrentReadoutEnvConfig, grid_size: int = 800) -> dict:
    u = np.linspace(0.0, env_cfg.horizon, grid_size)
    basis = np.exp(-np.outer(u, np.asarray(env_cfg.filter_decays, dtype=np.float64)))
    target = single_exp_kernel(u, env_cfg)
    signed_weights = _signed_lstsq(basis, target)
    signed_kernel = basis @ signed_weights
    positive_signed_kernel = np.maximum(signed_kernel, 0.0)
    return {
        "u": u,
        "target": target,
        "basis": basis,
        "signed_weights": signed_weights,
        "signed_kernel": signed_kernel,
        "positive_signed_kernel": positive_signed_kernel,
        "positive_signed_relative_l2": float(
            np.linalg.norm(positive_signed_kernel - target) / max(np.linalg.norm(target), 1e-12)
        ),
        "signed_relative_l2": float(np.linalg.norm(signed_kernel - target) / max(np.linalg.norm(target), 1e-12)),
    }


def stability_envelope_masses(env_cfg: CurrentReadoutEnvConfig) -> dict[str, float]:
    """Finite-horizon kernel masses entering Proposition 3.6."""

    global_fit = global_kernel_fit_weights(env_cfg, grid_size=1600)
    u = global_fit["u"]
    true_mass = env_cfg.excitation * (1.0 - np.exp(-env_cfg.decay * env_cfg.horizon)) / env_cfg.decay
    approximation_mass = env_cfg.excitation * np.trapz(np.abs(global_fit["signed_kernel"]), u)
    return {
        "true_envelope_mass": float(true_mass),
        "approximation_envelope_mass": float(approximation_mass),
    }


def fit_ct_memory_weights(
    env_cfg: CurrentReadoutEnvConfig,
    policy: Any,
    episodes: int = 40,
    seed: int = 5300,
) -> tuple[np.ndarray, dict]:
    """Fit an ex post signed readout from CT filters to the true memory.

    This diagnostic uses the known simulator intensity only after training. It
    is not part of CT-DDPG learning.
    """

    features = []
    targets = []
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        ep = collect_episode(env_cfg, policy, rng, exploration_std=0.0)
        states = ep.states[:-1]
        actions = ep.actions
        baseline = np.clip(
            env_cfg.base_intensity
            + env_cfg.state_intensity_sensitivity * states[:, 0]
            + env_cfg.action_intensity_sensitivity * actions,
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
    weights = _signed_lstsq(x, y)
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
    env_cfg: CurrentReadoutEnvConfig,
    policy: Any,
    save_path: str | Path | None = None,
    episodes: int = 40,
    seed: int = 5300,
):
    onpolicy_weights, summary = fit_ct_memory_weights(env_cfg, policy, episodes=episodes, seed=seed)
    global_fit = global_kernel_fit_weights(env_cfg)
    u = global_fit["u"]
    true_kernel = global_fit["target"]
    basis = global_fit["basis"]
    onpolicy_kernel = np.maximum(basis @ onpolicy_weights, 0.0)
    onpolicy_relative_l2 = float(
        np.linalg.norm(onpolicy_kernel - true_kernel) / max(np.linalg.norm(true_kernel), 1e-12)
    )
    summary.update(
        {
            "weights": global_fit["signed_weights"],
            "global_signed_weights": global_fit["signed_weights"],
            "kernel_relative_l2": global_fit["positive_signed_relative_l2"],
            "signed_kernel_relative_l2": global_fit["signed_relative_l2"],
            "onpolicy_kernel_relative_l2": onpolicy_relative_l2,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    axes[0].plot(u, true_kernel, label="true single-exponential kernel", linewidth=2.2)
    axes[0].plot(
        u,
        global_fit["positive_signed_kernel"],
        "--",
        label="positive part of global signed fit",
        linewidth=2.2,
    )
    axes[0].plot(u, onpolicy_kernel, ":", label="positive part of on-policy fit", linewidth=1.9)
    axes[0].set_xlabel("lag u")
    axes[0].set_ylabel("base kernel value")
    axes[0].set_title("Single-exponential kernel approximation")
    axes[0].set_xticks(np.linspace(0.0, env_cfg.horizon, 6))
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    labels = [f"{rho:.2f}" for rho in env_cfg.filter_decays]
    x_pos = np.arange(len(labels))
    axes[1].bar(x_pos - 0.2, global_fit["signed_weights"], width=0.4, color="#2563eb", label="global signed fit")
    axes[1].bar(x_pos + 0.2, onpolicy_weights, width=0.4, color="#16a34a", label="on-policy signed fit")
    axes[1].axhline(0.0, color="#111827", linewidth=0.8)
    max_tick_labels = 5
    tick_step = max(1, int(np.ceil(len(labels) / max_tick_labels)))
    tick_idx = np.arange(0, len(labels), tick_step)
    if tick_idx[-1] != len(labels) - 1:
        tick_idx = np.append(tick_idx, len(labels) - 1)
    axes[1].set_xticks(x_pos[tick_idx], [labels[i] for i in tick_idx])
    axes[1].set_xlabel("filter decay")
    axes[1].set_ylabel("signed coefficient")
    axes[1].set_title(
        f"global rel L2={summary['kernel_relative_l2']:.3g}, on-policy rel L2={onpolicy_relative_l2:.3g}"
    )
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=160)
    return fig, summary
