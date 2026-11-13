from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from erlang_env import ErlangHawkesConfig


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
    axes[0, 1].plot(episodes, [row.get("critic_loss", np.nan) for row in history], marker="o", label="critic loss")
    axes[0, 1].plot(episodes, [row.get("td_loss", np.nan) for row in history], marker="o", label="TD loss")
    axes[0, 1].plot(episodes, [row.get("terminal_loss", np.nan) for row in history], marker="o", label="terminal loss")
    axes[0, 1].set_title("Critic residual losses")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend()
    axes[1, 0].plot(episodes, [row.get("mean_abs_td", np.nan) for row in history], marker="o", label="mean |TD residual|")
    axes[1, 0].plot(
        episodes,
        [row.get("advantage_rate_abs", np.nan) for row in history],
        marker="o",
        label="off-policy |advantage rate|",
    )
    axes[1, 0].set_title("Bellman residual and replay advantage")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend()
    axes[1, 1].plot(episodes, [row.get("actor_grad_norm", np.nan) for row in history], marker="o", label="actor")
    axes[1, 1].plot(episodes, [row.get("action_grad_abs", np.nan) for row in history], marker="o", label="|dQ/da|")
    axes[1, 1].plot(episodes, [row.get("value_grad_norm", np.nan) for row in history], marker="o", label="value")
    axes[1, 1].plot(episodes, [row.get("qbar_grad_norm", np.nan) for row in history], marker="o", label="qbar")
    axes[1, 1].set_title("Gradient diagnostics")
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
    axes[0].set_yscale("log")
    axes[0].set_title("DGM/HJB convergence")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(steps, [row["mean_action"] for row in history], marker="o")
    axes[1].set_title("DGM argmin action in sampled domain")
    axes[1].grid(True, alpha=0.25)
    if save_path is not None:
        fig.savefig(_ensure_parent(save_path), dpi=180)
    return fig


def plot_benchmark_returns(rows: dict[str, dict], save_path: str | Path | None = None):
    reference_group = [
        key
        for key in ("dgm_hjb_oracle", *sorted(k for k in rows if k.startswith("constant_")))
        if key in rows
    ]
    groups = [
        reference_group,
        [key for key in ("ct_ddpg_current", "ct_ddpg_history", "ct_ddpg_filters", "ct_ddpg") if key in rows],
        [key for key in ("sac_current", "sac_history", "sac_filters", "sac") if key in rows],
        [key for key in ("ddpg_current", "ddpg_history", "ddpg_filters", "ddpg") if key in rows],
    ]
    grouped = {key for group in groups for key in group}
    extras = [key for key in rows if key not in grouped]
    if extras:
        groups.append(extras)

    labels = []
    positions = []
    cursor = 0.0
    for group in groups:
        if not group:
            continue
        labels.extend(group)
        positions.extend(cursor + np.arange(len(group), dtype=np.float64))
        cursor += len(group) + 0.8

    # The simulator stores reward-to-go, while the paper reports the equivalent
    # positive cost. Plot the latter so lower bars directly mean better policies.
    means = [-rows[key]["return_mean"] for key in labels]
    errs = [rows[key].get("return_ci90", 0.0) for key in labels]
    palette = {
        "dgm_hjb_oracle": "#dc2626",
        "ct_ddpg_current": "#93c5fd",
        "ct_ddpg_history": "#3b82f6",
        "ct_ddpg_filters": "#1d4ed8",
        "ct_ddpg": "#1d4ed8",
        "sac_current": "#a5f3fc",
        "sac_history": "#67c5d8",
        "sac_filters": "#0e7490",
        "sac": "#67c5d8",
        "ddpg_current": "#ddd6fe",
        "ddpg_history": "#a78bfa",
        "ddpg_filters": "#6d28d9",
        "ddpg": "#a78bfa",
    }
    display = {
        "dgm_hjb_oracle": "DGM/HJB\noracle",
        "ct_ddpg_current": "CT-DDPG\ncurrent",
        "ct_ddpg_history": "CT-DDPG\nhistory",
        "ct_ddpg_filters": "CT-DDPG\nfiltered",
        "ct_ddpg": "CT-DDPG\nfiltered",
        "sac_current": "SAC\ncurrent",
        "sac_history": "SAC\nhistory",
        "sac": "SAC\nhistory",
        "sac_filters": "SAC\nfiltered",
        "ddpg_current": "DDPG\ncurrent",
        "ddpg_history": "DDPG\nhistory",
        "ddpg": "DDPG\nhistory",
        "ddpg_filters": "DDPG\nfiltered",
    }
    tick_labels = [
        display.get(key, f"static\na={key.split('_', 1)[1]}" if key.startswith("constant_") else key)
        for key in labels
    ]
    colors = [palette.get(key, "#64748b") for key in labels]
    fig, ax = plt.subplots(figsize=(16, 5.8), constrained_layout=True)
    ax.bar(positions, means, yerr=errs, width=0.78, color=colors, alpha=0.9, capsize=4)
    ct_reference_key = "ct_ddpg_filters" if "ct_ddpg_filters" in rows else "ct_ddpg"
    if ct_reference_key in rows:
        ax.axhline(
            -rows[ct_reference_key]["return_mean"],
            color="#1d4ed8",
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            label="CT-DDPG filtered mean",
            zorder=4,
        )
        ax.legend(loc="upper left", frameon=False)
    ax.set_title("Erlang-kernel policy benchmark costs, 90% CI")
    ax.set_ylabel("discounted cost, lower is better")
    ax.set_xticks(positions, tick_labels)
    ax.tick_params(axis="x", rotation=0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.25)

    if save_path is not None:
        fig.savefig(_ensure_parent(save_path), dpi=180)
    return fig


def plot_observation_mode_costs(rows: dict[str, dict], save_path: str | Path | None = None):
    method_pairs = (
        ("ct_ddpg", "CT-DDPG", "#2563eb"),
        ("sac", "SAC", "#0e7490"),
        ("ddpg", "DDPG", "#6d28d9"),
    )
    centers = np.arange(len(method_pairs), dtype=np.float64)
    offset = 0.17
    plotted_means = []
    plotted_errs = []

    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    for method_index, (prefix, _, color) in enumerate(method_pairs):
        for mode, x_shift, filled in (("current", -offset, False), ("filters", offset, True)):
            key = f"{prefix}_{mode}"
            if key not in rows:
                continue
            mean = -float(rows[key]["return_mean"])
            err = float(rows[key].get("return_ci90", 0.0))
            x_pos = centers[method_index] + x_shift
            ax.errorbar(
                x_pos,
                mean,
                yerr=err,
                fmt="o",
                markersize=10,
                capsize=6,
                linewidth=2,
                color=color,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=2,
                label=mode if method_index == 0 else "_nolegend_",
                zorder=3,
            )
            ax.annotate(
                f"{mean:.3f}",
                (x_pos, mean),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                fontweight="semibold",
            )
            plotted_means.append(mean)
            plotted_errs.append(err)

    ct_reference_key = "ct_ddpg_filters" if "ct_ddpg_filters" in rows else "ct_ddpg"
    if ct_reference_key in rows:
        ax.axhline(
            -rows[ct_reference_key]["return_mean"],
            color="#1d4ed8",
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            label="CT-DDPG filtered mean",
            zorder=1,
        )

    means_arr = np.asarray(plotted_means)
    errs_arr = np.asarray(plotted_errs)
    lower = float(np.min(means_arr - errs_arr))
    upper = float(np.max(means_arr + errs_arr))
    padding = max(0.01, 0.10 * max(upper - lower, 0.02))
    ax.set_ylim(max(0.0, lower - padding), upper + padding)
    ax.set_xlim(-0.55, len(method_pairs) - 0.45)
    ax.set_xticks(centers, [label for _, label, _ in method_pairs])
    ax.set_title("Erlang current versus filtered costs, expanded scale")
    ax.set_ylabel("discounted cost, lower is better")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, ncols=3)
    if save_path is not None:
        fig.savefig(_ensure_parent(save_path), dpi=180)
    return fig


def plot_filtered_training_budget(
    rows: dict[str, dict],
    training_transitions: dict[str, int],
    save_path: str | Path | None = None,
):
    """Compare final common-test cost with each filtered policy's full sample budget."""

    series = (
        ("ct_ddpg_filters", "CT-DDPG filtered", "#1d4ed8", "o"),
        ("sac_filters", "SAC filtered", "#0e7490", "s"),
        ("ddpg_filters", "DDPG filtered", "#6d28d9", "^"),
    )
    available = [item for item in series if item[0] in rows and item[0] in training_transitions]
    if not available:
        raise ValueError("no filtered policy has both a test result and a training budget")

    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    for key, label, color, marker in available:
        transitions = int(training_transitions[key])
        cost = -float(rows[key]["return_mean"])
        ci90 = float(rows[key].get("return_ci90", 0.0))
        ax.errorbar(
            transitions,
            cost,
            yerr=ci90,
            fmt=marker,
            markersize=11,
            capsize=6,
            linewidth=2,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
            zorder=3,
        )
        ax.annotate(
            f"{cost:.3f} at {transitions / 1000:.0f}k",
            (transitions, cost),
            xytext=(8, 9),
            textcoords="offset points",
            color=color,
            fontsize=10,
            fontweight="semibold",
        )

    ax.set_xscale("log")
    ax.set_xlabel("training environment transitions (log scale)")
    ax.set_ylabel("discounted test cost, lower is better")
    ax.set_title("Filtered-policy cost versus full training budget")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    if save_path is not None:
        fig.savefig(_ensure_parent(save_path), dpi=180)
    return fig


def plot_sb3_training_histories(
    metadata: dict[str, dict],
    save_path: str | Path | None = None,
):
    """Plot checkpoint cost estimates for the tuned SAC/DDPG observation variants."""

    colors = {
        "sac_current": "#a5f3fc",
        "sac_history": "#67c5d8",
        "sac_filters": "#0e7490",
        "ddpg_current": "#ddd6fe",
        "ddpg_history": "#a78bfa",
        "ddpg_filters": "#6d28d9",
    }
    labels = {
        "sac_current": "SAC current",
        "sac_history": "SAC history",
        "sac_filters": "SAC filtered",
        "ddpg_current": "DDPG current",
        "ddpg_history": "DDPG history",
        "ddpg_filters": "DDPG filtered",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False, constrained_layout=True)
    for ax, algorithm in zip(axes, ("sac", "ddpg")):
        for key in (f"{algorithm}_current", f"{algorithm}_history", f"{algorithm}_filters"):
            history = metadata.get(key, {}).get("training_history", [])
            if not history:
                continue
            steps = np.asarray([row["timesteps"] for row in history], dtype=np.float64)
            costs = np.asarray([row["cost_mean"] for row in history], dtype=np.float64)
            running_best = np.minimum.accumulate(costs)
            best_idx = int(np.argmin(costs))
            ax.plot(
                steps,
                costs,
                marker=".",
                markersize=3,
                linewidth=1,
                color=colors[key],
                alpha=0.28,
            )
            ax.plot(
                steps,
                running_best,
                marker="o",
                markersize=4,
                linewidth=2.2,
                color=colors[key],
                label=labels[key],
            )
            ax.scatter(
                steps[best_idx],
                costs[best_idx],
                marker="*",
                s=115,
                color=colors[key],
                edgecolor="black",
                linewidth=0.6,
                zorder=5,
            )
        ax.set_title(f"{algorithm.upper()} validation checkpoint convergence")
        ax.set_xlabel("environment transitions")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    for ax in axes:
        ax.set_ylabel("checkpoint mean cost (lower is better)")
    if save_path is not None:
        fig.savefig(_ensure_parent(save_path), dpi=180)
    return fig


def plot_value_surface(
    value_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    env_cfg: ErlangHawkesConfig,
    save_path: str | Path | None = None,
    title: str = "Cost-to-go surface",
    z_label: str = "estimated cost",
    y_level: float = 0.0,
    grid_size: int = 39,
):
    t_grid = np.linspace(0.0, env_cfg.horizon, grid_size)
    x_grid = np.linspace(-0.8, 1.5, grid_size)
    tt, xx = np.meshgrid(t_grid, x_grid)
    states = np.zeros((xx.size, env_cfg.state_dim), dtype=np.float64)
    states[:, 0] = xx.reshape(-1)
    states[:, 1] = y_level
    states[:, 2] = y_level
    values = value_fn(tt.reshape(-1), states).reshape(xx.shape)
    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(tt, xx, values, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("X")
    ax.set_zlabel(z_label)
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
    if save_path is not None:
        fig.savefig(_ensure_parent(save_path), dpi=180)
    return fig


def plot_policy_mc_cost_surface(
    policy,
    env_cfg: ErlangHawkesConfig,
    save_path: str | Path | None = None,
    title: str = "Monte Carlo policy cost surface",
    y_level: float = 0.0,
    z_level: float = 0.0,
    grid_size: int = 25,
    mc_episodes: int = 4,
    seed: int = 2027,
):
    """Estimate cost-to-go by rolling out a policy from grid states.

    This is slower than plotting the raw critic, but it is the meaningful
    comparison to the HJB/DGM cost surface for a model-free policy.
    """

    from ct_ddpg_erlang import collect_episode_from_state

    t_grid = np.linspace(0.0, env_cfg.horizon, grid_size)
    x_grid = np.linspace(-0.8, 1.5, grid_size)
    tt, xx = np.meshgrid(t_grid, x_grid)
    values = np.zeros_like(xx, dtype=np.float64)
    base_state = np.zeros(env_cfg.state_dim, dtype=np.float64)
    base_state[1] = y_level
    base_state[2] = y_level
    base_state[3:] = z_level
    for idx, (time, x_value) in enumerate(zip(tt.reshape(-1), xx.reshape(-1))):
        state = base_state.copy()
        state[0] = x_value
        costs = []
        for rep in range(mc_episodes):
            rng = np.random.default_rng(seed + 10_000 * idx + rep)
            episode = collect_episode_from_state(env_cfg, policy, rng, time, state)
            costs.append(-episode.discounted_return)
        values.reshape(-1)[idx] = float(np.mean(costs))

    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(tt, xx, values, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("X")
    ax.set_zlabel("Monte Carlo cost")
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
    if save_path is not None:
        fig.savefig(_ensure_parent(save_path), dpi=180)
    return fig


def realized_remaining_cost(episode, env_cfg: ErlangHawkesConfig) -> np.ndarray:
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
        "CT-DDPG current": "#93c5fd",
        "CT-DDPG history": "#3b82f6",
        "CT-DDPG filtered": "#1d4ed8",
        "DGM/HJB oracle": "#dc2626",
        "SAC": "#0891b2",
        "SAC current": "#a5f3fc",
        "SAC history": "#67c5d8",
        "SAC filtered": "#0e7490",
        "DDPG": "#7c3aed",
        "DDPG current": "#ddd6fe",
        "DDPG history": "#a78bfa",
        "DDPG filtered": "#6d28d9",
        "PPO": "#f59e0b",
        "SAC_FILTERS": "#0e7490",
        "DDPG_FILTERS": "#6d28d9",
        "PPO_FILTERS": "#d97706",
    }
    linestyles = {
        "CT-DDPG": "-",
        "CT-DDPG current": ":",
        "CT-DDPG history": "--",
        "CT-DDPG filtered": "-",
        "DGM/HJB oracle": "--",
        "SAC": "-.",
        "SAC current": ":",
        "SAC history": "--",
        "SAC filtered": "-",
        "DDPG": (0, (3, 1, 1, 1)),
        "DDPG current": ":",
        "DDPG history": "--",
        "DDPG filtered": "-",
        "PPO": (0, (5, 2)),
        "SAC_FILTERS": "-.",
        "DDPG_FILTERS": (0, (3, 1, 1, 1)),
        "PPO_FILTERS": (0, (5, 2)),
    }
    fig, axes = plt.subplots(6, 1, figsize=(14, 17), sharex=True, constrained_layout=True)
    first = next(iter(next(iter(episode_groups.values()))))
    times = first.times
    step_times = times[:-1]
    physical_state_dim = min(
        ep.states.shape[1]
        for episodes in episode_groups.values()
        for ep in episodes
    )
    for label, episodes in episode_groups.items():
        color = colors.get(label, "#64748b")
        ls = linestyles.get(label, ":")
        stacks = [
            (axes[0], times, np.stack([ep.states[:, 0] for ep in episodes])),
            (axes[1], times, np.stack([ep.states[:, 2] for ep in episodes])),
            (axes[3], step_times, np.stack([ep.actions for ep in episodes])),
            (axes[4], step_times, np.stack([ep.intensities for ep in episodes])),
        ]
        if "filtered" in label.lower():
            stacks.append(
                (
                    axes[2],
                    times,
                    np.stack(
                        [np.linalg.norm(ep.states[:, 3:physical_state_dim], axis=1) for ep in episodes]
                    ),
                )
            )
        for ax, xs, vals in stacks:
            mean, lo, hi = _ci_band(vals, ci=ci)
            ax.plot(xs, mean, color=color, linestyle=ls, linewidth=2.2, label=label)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.14, linewidth=0)
        if value_path_groups is not None and label in value_path_groups:
            mean, lo, hi = _ci_band(np.stack(value_path_groups[label]), ci=ci)
            axes[5].plot(times, mean, color=color, linestyle=ls, linewidth=2.2, label=label)
            axes[5].fill_between(times, lo, hi, color=color, alpha=0.14, linewidth=0)
    titles = [
        f"State path X_t: mean with {int(ci * 100)}% CI",
        "Exact Erlang memory phase Y2_t",
        "Model-free exponential filter norm ||Z_t||",
        "Control path a_t",
        "Intensity path lambda_t",
        "Realized remaining discounted cost, lower is better",
    ]
    ylabels = ["X", "Y2", "||Z||", "a", "lambda", "remaining cost"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time")
    if save_path is not None:
        fig.savefig(_ensure_parent(save_path), dpi=180)
    return fig
