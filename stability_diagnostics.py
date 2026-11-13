from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np


def uncapped_path_diagnostics(
    env_factory: Callable[[], object],
    actions: Iterable[float] = (0.0, 0.3, 1.0),
    episodes: int = 40,
    seed: int = 8800,
) -> dict:
    """Stress-test the actual uncapped simulator under constant controls.

    The diagnostic never truncates the intensity or event count. It is a
    numerical sanity check, not the proof of nonexplosion; the parameter-based
    subcriticality certificate supplies the latter.
    """

    rng = np.random.default_rng(seed)
    raw: dict[str, dict[str, list[float]]] = {}
    for action in actions:
        key = f"{float(action):.2f}"
        raw[key] = {
            "event_counts": [],
            "max_intensities": [],
            "integrated_intensities": [],
            "max_abs_states": [],
        }
        for _ in range(int(episodes)):
            env = env_factory()
            state = np.asarray(env.reset(), dtype=np.float64)
            total_events = 0
            max_intensity = 0.0
            integrated_intensity = 0.0
            max_abs_state = abs(float(state[0]))
            done = False
            while not done:
                state, _, done, info = env.step(float(action), rng)
                intensity = float(info["lambda"])
                if not np.isfinite(intensity) or intensity < 0.0:
                    raise RuntimeError(f"Invalid uncapped intensity {intensity} at action {action}")
                total_events += int(info["jumps"])
                max_intensity = max(max_intensity, intensity)
                integrated_intensity += intensity * float(env.cfg.dt)
                max_abs_state = max(max_abs_state, abs(float(state[0])))
            raw[key]["event_counts"].append(float(total_events))
            raw[key]["max_intensities"].append(max_intensity)
            raw[key]["integrated_intensities"].append(integrated_intensity)
            raw[key]["max_abs_states"].append(max_abs_state)

    summary: dict[str, dict[str, float] | dict] = {"raw": raw}
    for key, values in raw.items():
        summary[key] = {}
        for metric, samples in values.items():
            arr = np.asarray(samples, dtype=np.float64)
            summary[key][f"{metric}_mean"] = float(np.mean(arr))
            summary[key][f"{metric}_q95"] = float(np.quantile(arr, 0.95))
            summary[key][f"{metric}_max"] = float(np.max(arr))
    return summary


def sufficient_condition_report(
    *,
    true_envelope_mass: float,
    approximation_envelope_mass: float,
    baseline_bound: float,
    jump_amplitude_bound: float,
    p_star: float = 4.0,
) -> dict[str, float | bool | str]:
    """Return the implemented Proposition 3.6 checklist.

    Compact control space, deterministic finite initial state, and globally
    Lipschitz/clipped coefficient maps are properties of all three experiment
    classes. The two envelope masses are supplied by each kernel diagnostic.
    """

    rho_t = max(float(true_envelope_mass), float(approximation_envelope_mass))
    return {
        "p_star": float(p_star),
        "finite_initial_moment": True,
        "compact_action_space": "[0, 1]",
        "globally_lipschitz_coefficients": True,
        "uniform_baseline_bound": float(baseline_bound),
        "uniform_jump_amplitude_bound": float(jump_amplitude_bound),
        "true_kernel_envelope_mass": float(true_envelope_mass),
        "approximation_kernel_envelope_mass": float(approximation_envelope_mass),
        "rho_T": rho_t,
        "subcritical": bool(rho_t < 1.0),
        "assumption_3_2_certified_by_proposition_3_6": bool(rho_t < 1.0),
    }


def plot_stability_certificate(
    report: dict,
    path_summary: dict,
    save_path: str | Path | None = None,
    title: str = "Assumption 3.2 sufficient-condition diagnostic",
):
    action_keys = sorted(path_summary["raw"], key=float)
    event_samples = [path_summary["raw"][key]["event_counts"] for key in action_keys]
    intensity_samples = [path_summary["raw"][key]["max_intensities"] for key in action_keys]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    masses = [report["true_kernel_envelope_mass"], report["approximation_kernel_envelope_mass"]]
    axes[0].bar(["true", "finite approximation"], masses, color=["#2563eb", "#16a34a"])
    axes[0].axhline(1.0, color="#b91c1c", linestyle="--", linewidth=1.4, label=r"subcritical threshold $1$")
    axes[0].set_ylabel(r"finite-horizon envelope mass $\rho_T$")
    axes[0].set_title("Analytical certificate")
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].boxplot(event_samples, tick_labels=[f"a={key}" for key in action_keys], showfliers=False)
    axes[1].set_ylabel("events over the horizon")
    axes[1].set_title("Uncapped event counts")
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].boxplot(intensity_samples, tick_labels=[f"a={key}" for key in action_keys], showfliers=False)
    axes[2].set_ylabel("pathwise maximum intensity")
    axes[2].set_title("Uncapped intensity stress test")
    axes[2].grid(True, axis="y", alpha=0.25)

    fig.suptitle(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=170)
    return fig

