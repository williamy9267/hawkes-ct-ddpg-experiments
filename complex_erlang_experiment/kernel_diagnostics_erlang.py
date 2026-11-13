from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ct_ddpg_erlang import collect_episode
from erlang_env import ErlangHawkesConfig, readout_multiplier


def _signed_lstsq(features: np.ndarray, target: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    lhs = features.T @ features + ridge * np.eye(features.shape[1])
    rhs = features.T @ target
    return np.linalg.solve(lhs, rhs).astype(np.float64)


def erlang_kernel(u: np.ndarray, cfg: ErlangHawkesConfig) -> np.ndarray:
    u_arr = np.asarray(u, dtype=np.float64)
    rho = cfg.erlang_decay
    return rho * u_arr * np.exp(-rho * u_arr)


def global_kernel_fit_weights(
    env_cfg: ErlangHawkesConfig,
    grid_size: int = 1200,
) -> dict:
    """Fit the Erlang kernel itself on a deterministic lag grid.

    This is the clean approximation-theory diagnostic. The on-policy memory fit
    below is useful, but it can be more sensitive to path distribution and
    notebook cache/state.
    """

    u = np.linspace(0.0, env_cfg.horizon, grid_size)
    basis = np.exp(-np.outer(u, np.asarray(env_cfg.filter_decays, dtype=np.float64)))
    target = erlang_kernel(u, env_cfg)
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


def stability_envelope_masses(env_cfg: ErlangHawkesConfig) -> dict[str, float]:
    """Finite-horizon true and signed-approximation envelope masses."""

    global_fit = global_kernel_fit_weights(env_cfg, grid_size=2000)
    u = global_fit["u"]
    rho = env_cfg.erlang_decay
    true_mass = env_cfg.excitation * (1.0 - np.exp(-rho * env_cfg.horizon) * (1.0 + rho * env_cfg.horizon)) / rho
    approximation_mass = env_cfg.excitation * np.trapz(np.abs(global_fit["signed_kernel"]), u)
    return {
        "true_envelope_mass": float(true_mass),
        "approximation_envelope_mass": float(approximation_mass),
    }


def fit_ct_memory_weights(
    env_cfg: ErlangHawkesConfig,
    policy: Any,
    episodes: int = 40,
    seed: int = 6300,
) -> tuple[np.ndarray, dict]:
    """Fit an ex post signed readout from CT filters to Erlang memory."""

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
            features.append(states[mask, 3:])
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
    env_cfg: ErlangHawkesConfig,
    policy: Any,
    save_path: str | Path | None = None,
    episodes: int = 40,
    seed: int = 6300,
):
    onpolicy_weights, summary = fit_ct_memory_weights(env_cfg, policy, episodes=episodes, seed=seed)
    global_fit = global_kernel_fit_weights(env_cfg, grid_size=1200)
    u = global_fit["u"]
    true_kernel = global_fit["target"]
    basis = global_fit["basis"]
    onpolicy_signed_kernel = basis @ onpolicy_weights
    onpolicy_positive_kernel = np.maximum(onpolicy_signed_kernel, 0.0)
    onpolicy_relative_l2 = float(
        np.linalg.norm(onpolicy_positive_kernel - true_kernel) / max(np.linalg.norm(true_kernel), 1e-12)
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(u, true_kernel, label="true Erlang-2 kernel", linewidth=2.2)
    axes[0].plot(
        u,
        global_fit["positive_signed_kernel"],
        "--",
        label="positive part of global signed fit",
        linewidth=2.2,
    )
    axes[0].plot(u, onpolicy_positive_kernel, ":", label="positive part of on-policy fit", linewidth=1.9)
    axes[0].set_xlabel("lag u")
    axes[0].set_ylabel("base kernel value")
    axes[0].set_title("Erlang kernel approximation by exponential dictionary")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    labels = [f"{eta:.2f}" for eta in env_cfg.filter_decays]
    x_pos = np.arange(len(labels))
    axes[1].bar(x_pos - 0.2, global_fit["signed_weights"], width=0.4, color="#2563eb", label="global signed fit")
    axes[1].bar(x_pos + 0.2, onpolicy_weights, width=0.4, color="#16a34a", label="on-policy signed fit")
    axes[1].axhline(0.0, color="#111827", linewidth=0.8)
    axes[1].set_xticks(x_pos, labels)
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
