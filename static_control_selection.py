from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np


def _normalized(value):
    return json.loads(json.dumps(value, sort_keys=True))


def select_static_control(
    env_cfg,
    evaluate: Callable,
    actions: Iterable[float],
    *,
    seed: int,
    episodes: int,
    cache_path: str | Path,
    environment_signature: dict,
    projection_step: float = 0.01,
    force: bool = False,
) -> dict:
    """Select a constant action on validation paths, independently of final testing.

    A quadratic smooths the Monte Carlo profile locally. Its minimizer is
    projected to the declared action grid before the policy is frozen.
    """

    action_grid = np.asarray(tuple(actions), dtype=np.float64)
    if action_grid.ndim != 1 or action_grid.size < 3:
        raise ValueError("actions must contain at least three one-dimensional grid points")
    if not np.all(np.diff(action_grid) > 0.0):
        raise ValueError("actions must be strictly increasing")

    cache_path = Path(cache_path)
    signature = {
        "environment": _normalized(environment_signature),
        "actions": [float(x) for x in action_grid],
        "seed": int(seed),
        "episodes": int(episodes),
        "projection_step": float(projection_step),
        "method": "local_quadratic_profile_projected_to_grid",
    }
    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("signature") == signature:
            return cached

    rows = []
    for action in action_grid:
        stats = evaluate(
            env_cfg,
            None,
            seed=seed,
            episodes=episodes,
            constant_action=float(action),
        )
        rows.append(
            {
                "action": float(action),
                "cost": float(-stats["return_mean"]),
                "cost_ci90": float(stats["return_ci90"]),
                "jumps_mean": float(stats["jumps_mean"]),
                "intensity_mean": float(stats["intensity_mean"]),
            }
        )

    costs = np.asarray([row["cost"] for row in rows], dtype=np.float64)
    coefficients = np.polyfit(action_grid, costs, deg=2)
    if coefficients[0] > 0.0:
        continuous_action = -coefficients[1] / (2.0 * coefficients[0])
        continuous_action = float(np.clip(continuous_action, action_grid[0], action_grid[-1]))
    else:
        continuous_action = float(action_grid[int(np.argmin(costs))])
    selected_action = projection_step * round(continuous_action / projection_step)
    selected_action = float(np.clip(selected_action, action_grid[0], action_grid[-1]))

    result = {
        "signature": signature,
        "profile": rows,
        "quadratic_coefficients": [float(x) for x in coefficients],
        "continuous_minimizer": continuous_action,
        "selected_action": selected_action,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def plot_static_control_selection(
    selection: dict,
    save_path: str | Path | None = None,
    *,
    title: str = "Validation selection of the static control",
):
    rows = selection["profile"]
    actions = np.asarray([row["action"] for row in rows], dtype=np.float64)
    costs = np.asarray([row["cost"] for row in rows], dtype=np.float64)
    intervals = np.asarray([row["cost_ci90"] for row in rows], dtype=np.float64)
    coefficients = np.asarray(selection["quadratic_coefficients"], dtype=np.float64)
    selected = float(selection["selected_action"])

    dense_actions = np.linspace(actions.min(), actions.max(), 240)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.errorbar(
        actions,
        costs,
        yerr=intervals,
        fmt="o",
        color="#2563eb",
        ecolor="#93c5fd",
        capsize=3,
        label="validation estimate, 90% CI",
    )
    ax.plot(dense_actions, np.polyval(coefficients, dense_actions), color="#111827", label="quadratic profile")
    ax.axvline(selected, color="#dc2626", linestyle="--", linewidth=1.5, label=fr"selected $a^*={selected:.2f}$")
    ax.set_xlabel("constant control level")
    ax.set_ylabel("discounted validation cost")
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
    return fig
