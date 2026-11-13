from __future__ import annotations

import base64
from pathlib import Path

import matplotlib
import nbformat
import numpy as np

matplotlib.use("Agg")

from ct_ddpg_powerlaw import collect_episode
from dgm_powerlaw_oracle import DGMOraclePolicy, load_dgm_oracle
from plotting_powerlaw import plot_episode_ensemble_comparison, realized_remaining_cost
from powerlaw_env import PowerLawHawkesConfig
from sb3_baselines_powerlaw import collect_sb3_episode
from train_tuned_ct_powerlaw_suite import load_tuned_ct_powerlaw_suite
from train_tuned_sb3_powerlaw_suite import load_tuned_powerlaw_suite


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results"
FIGURE_PATH = HERE / "figures" / "policy_path_ensemble_ci.png"
NOTEBOOK_PATH = HERE / "powerlaw_kernel_comparison.ipynb"


def main() -> None:
    env_cfg = PowerLawHawkesConfig()
    ct_agents, _, _ = load_tuned_ct_powerlaw_suite(RESULT_DIR)
    dgm_oracle, _ = load_dgm_oracle(RESULT_DIR / "dgm_oracle.pt", device="cpu")
    dgm_policy = DGMOraclePolicy(dgm_oracle)
    sb3_models = load_tuned_powerlaw_suite(RESULT_DIR)

    static_action = 0.34
    episode_groups = {f"CT-DDPG {mode if mode != 'filters' else 'filtered'}": [] for mode in ct_agents}
    episode_groups.update({"DGM/HJB oracle": [], f"static a={static_action:.2f}": []})
    for key, (_, mode, _, _) in sb3_models.items():
        episode_groups[f"{key.split('_', 1)[0].upper()} {'filtered' if mode == 'filters' else mode}"] = []
    value_path_groups = {key: [] for key in episode_groups}

    for path_index in range(20):
        seed = 9000 + path_index
        for mode, agent in ct_agents.items():
            label = f"CT-DDPG {'filtered' if mode == 'filters' else mode}"
            episode = collect_episode(env_cfg, agent, np.random.default_rng(seed), exploration_std=0.0)
            episode_groups[label].append(episode)
            value_path_groups[label].append(realized_remaining_cost(episode, env_cfg))

        dgm_episode = collect_episode(env_cfg, dgm_policy, np.random.default_rng(seed), exploration_std=0.0)
        static_episode = collect_episode(
            env_cfg,
            None,
            np.random.default_rng(seed),
            constant_action=static_action,
        )
        episode_groups["DGM/HJB oracle"].append(dgm_episode)
        episode_groups[f"static a={static_action:.2f}"].append(static_episode)
        value_path_groups["DGM/HJB oracle"].append(realized_remaining_cost(dgm_episode, env_cfg))
        value_path_groups[f"static a={static_action:.2f}"].append(
            realized_remaining_cost(static_episode, env_cfg)
        )

        for key, (model, mode, history_len, feature_mode) in sb3_models.items():
            label = f"{key.split('_', 1)[0].upper()} {'filtered' if mode == 'filters' else mode}"
            episode = collect_sb3_episode(
                env_cfg,
                model,
                observation_mode=mode,
                history_len=history_len,
                seed=seed,
                feature_mode=feature_mode,
            )
            episode_groups[label].append(episode)
            value_path_groups[label].append(realized_remaining_cost(episode, env_cfg))

    plot_episode_ensemble_comparison(
        episode_groups,
        value_path_groups,
        FIGURE_PATH,
        ci=0.90,
        kernel_weights=env_cfg.weights,
    )

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    notebook.cells[12].outputs = [
        nbformat.v4.new_output(
            "display_data",
            data={"image/png": base64.b64encode(FIGURE_PATH.read_bytes()).decode("ascii")},
            metadata={},
        )
    ]
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()
