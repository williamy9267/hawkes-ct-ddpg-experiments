from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from current_readout_env import CurrentReadoutEnvConfig


def _ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _ci_band(values: np.ndarray, ci: float = 0.90):
    values = np.asarray(values, dtype=np.float64)
    mean = np.mean(values, axis=0)
    if values.shape[0] <= 1:
        return mean, mean, mean
    z = 1.6448536269514722 if abs(ci - 0.90) < 1e-12 else 1.96
    half = z * np.std(values, axis=0) / np.sqrt(values.shape[0])
    return mean, mean - half, mean + half


def plot_ct_history(history: list[dict], save_path: str | Path | None = None):
    episodes = [row["episode"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    axes[0, 0].plot(episodes, [row.get("eval_return_mean", np.nan) for row in history], marker="o")
    axes[0, 0].set_title("CT-DDPG validation return")
    axes[0, 0].set_xlabel("episode")
    axes[0, 0].set_ylabel("return")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(episodes, [row.get("critic_loss", np.nan) for row in history], marker="o", label="critic")
    axes[0, 1].plot(episodes, [row.get("actor_loss", np.nan) for row in history], marker="o", label="actor")
    axes[0, 1].set_title("Actor and critic losses")
    axes[0, 1].set_xlabel("episode")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].plot(episodes, [row.get("advantage_rate_abs", np.nan) for row in history], marker="o")
    axes[1, 0].set_title("Mean absolute advantage rate")
    axes[1, 0].set_xlabel("episode")
    axes[1, 0].set_ylabel("|A|")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(episodes, [row.get("actor_grad_norm", np.nan) for row in history], marker="o", label="actor")
    axes[1, 1].plot(episodes, [row.get("value_grad_norm", np.nan) for row in history], marker="o", label="value")
    axes[1, 1].plot(episodes, [row.get("qbar_grad_norm", np.nan) for row in history], marker="o", label="qbar")
    axes[1, 1].set_title("Gradient norms")
    axes[1, 1].set_xlabel("episode")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend()

    if save_path is not None:
        fig.savefig(_ensure_parent(save_path), dpi=180)
    return fig


def plot_dgm_history(history: list[dict], save_path: str | Path | None = None):
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(steps, [row["loss"] for row in history], marker="o", label="loss")
    axes[0].plot(steps, [row["residual_loss"] for row in history], marker="o", label="residual")
    axes[0].set_title("DGM/HJB convergence")
    axes[0].set_xlabel("step")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(steps, [row["mean_action"] for row in history], marker="o")
    axes[1].set_title("DGM argmin action in sampled domain")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("mean action")
    axes[1].grid(True, alpha=0.25)
    if save_path is not None:
        fig.savefig(_ensure_parent(save_path), dpi=180)
    return fig


def plot_benchmark_returns(rows: dict[str, dict], save_path: str | Path | None = None):
    preferred = ("dgm_hjb_oracle", "constant_0.30", "ct_ddpg")
    labels = [key for key in preferred if key in rows]
    labels.extend(key for key in rows if key not in labels)
    means = [-rows[key]["return_mean"] for key in labels]
    errs = [rows[key].get("return_ci90", 0.0) for key in labels]
    palette = {
        "dgm_hjb_oracle": "#dc2626",
        "ct_ddpg": "#1d4ed8",
    }
    display = {
        "dgm_hjb_oracle": "DGM/HJB\noracle",
        "ct_ddpg": "CT-DDPG",
    }
    tick_labels = [
        display.get(key, f"static\na={key.split('_', 1)[1]}" if key.startswith("constant_") else key)
        for key in labels
    ]
    colors = [palette.get(key, "#64748b") for key in labels]
    positions = np.arange(len(labels), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    ax.bar(positions, means, yerr=errs, width=0.72, color=colors, alpha=0.9, capsize=4)
    if "ct_ddpg" in rows:
        ax.axhline(
            -rows["ct_ddpg"]["return_mean"],
            color="#1d4ed8",
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            label="CT-DDPG mean",
            zorder=4,
        )
        ax.legend(loc="upper left", frameon=False)
    ax.set_title("Single-exponential policy benchmark costs, 90% CI")
    ax.set_ylabel("discounted cost, lower is better")
    ax.set_xticks(positions, tick_labels)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.25)
    if save_path is not None:
        fig.savefig(_ensure_parent(save_path), dpi=180)
    return fig


def plot_value_surface(
    value_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    env_cfg: CurrentReadoutEnvConfig,
    save_path: str | Path | None = None,
    title: str = "Cost-to-go surface",
    z_level: float = 0.0,
    grid_size: int = 45,
):
    t_grid = np.linspace(0.0, env_cfg.horizon, grid_size)
    x_grid = np.linspace(-0.8, 1.5, grid_size)
    tt, xx = np.meshgrid(t_grid, x_grid)
    states = np.zeros((xx.size, env_cfg.state_dim), dtype=np.float64)
    states[:, 0] = xx.reshape(-1)
    states[:, 1:] = z_level
    values = value_fn(tt.reshape(-1), states).reshape(xx.shape)

    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(tt, xx, values, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("X")
    ax.set_zlabel("estimated cost")
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
    if save_path is not None:
        fig.savefig(_ensure_parent(save_path), dpi=180)
    return fig


def realized_remaining_cost(episode, env_cfg: CurrentReadoutEnvConfig) -> np.ndarray:
    """Discounted cost-to-go along one realized path, comparable across policies."""
    costs = -np.asarray(episode.reward_rates, dtype=np.float64)
    out = np.zeros(len(episode.times), dtype=np.float64)
    running = env_cfg.terminal_x_cost * episode.states[-1, 0] ** 2
    out[-1] = running
    step_discount = float(np.exp(-env_cfg.beta * env_cfg.dt))
    for k in range(len(costs) - 1, -1, -1):
        running = costs[k] * env_cfg.dt + step_discount * running
        out[k] = running
    return out


def plot_episode_ensemble_comparison(
    episode_groups: dict[str, list],
    value_path_groups: dict[str, list[np.ndarray]] | None,
    save_path: str | Path | None = None,
    ci: float = 0.90,
):
    colors = {
        "CT-DDPG": "#2563eb",
        "DGM/HJB oracle": "#dc2626",
    }
    linestyles = {
        "CT-DDPG": "-",
        "DGM/HJB oracle": "--",
    }
    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True, constrained_layout=True)

    first = next(iter(next(iter(episode_groups.values()))))
    times = first.times
    step_times = times[:-1]

    for label, episodes in episode_groups.items():
        color = colors.get(label, "#64748b")
        ls = linestyles.get(label, ":")
        x_stack = np.stack([ep.states[:, 0] for ep in episodes])
        z_stack = np.stack([np.linalg.norm(ep.states[:, 1:], axis=1) for ep in episodes])
        action_stack = np.stack([ep.actions for ep in episodes])
        intensity_stack = np.stack([ep.intensities for ep in episodes])

        for ax, xs, xvals in [
            (axes[0], times, x_stack),
            (axes[1], times, z_stack),
            (axes[2], step_times, action_stack),
            (axes[3], step_times, intensity_stack),
        ]:
            mean, lo, hi = _ci_band(xvals, ci=ci)
            ax.plot(xs, mean, color=color, linestyle=ls, linewidth=2.4, label=label)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.14, linewidth=0)

        if value_path_groups is not None and label in value_path_groups:
            v_stack = np.stack(value_path_groups[label])
            mean, lo, hi = _ci_band(v_stack, ci=ci)
            axes[4].plot(times, mean, color=color, linestyle=ls, linewidth=2.4, label=label)
            axes[4].fill_between(times, lo, hi, color=color, alpha=0.14, linewidth=0)

    axes[0].set_title(f"State path X_t: mean with {int(ci * 100)}% CI over {len(next(iter(episode_groups.values())))} paths")
    axes[0].set_ylabel("X")
    axes[1].set_title("Online Markov memory norm ||Z_t||")
    axes[1].set_ylabel("||Z||")
    axes[2].set_title("Control path a_t")
    axes[2].set_ylabel("a")
    axes[3].set_title("Intensity path lambda_t")
    axes[3].set_ylabel("lambda")
    axes[4].set_title("Realized remaining discounted cost, lower is better")
    axes[4].set_ylabel("remaining cost")
    axes[4].set_xlabel("time")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    if save_path is not None:
        fig.savefig(_ensure_parent(save_path), dpi=180)
    return fig
