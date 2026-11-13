from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODES = ("current", "filters")

DEFAULT_TRAINING = {
    "episodes": 180,
    "warmup_episodes": 30,
    "updates_per_episode": 10,
    "eval_every": 20,
    "eval_episodes": 120,
    "checkpoint_seed": 4019,
    "early_stop_patience": 6,
    "validation_episodes": 320,
    "validation_seed": 8100,
    "history_len": 24,
    "seed": 19,
    "blas_threads": 2,
    "lr_actor": 5e-6,
    "lr_actor_profile": 0.0,
    "actor_lr_final_scale": 1.0,
    "lr_critic": 6e-4,
    "tau": 0.003,
    "initial_action": 0.18,
    "actor_t_skip": 0.0,
    "actor_x_skip": 2.0,
    "actor_z_skip": 0.0,
    "actor_z_slope_skip": 0.0,
    "actor_z_curve_skip": 0.0,
    "actor_bias_shift": 0.0,
    "actor_update_every": 1,
}

DEFAULT_TRAINING_BY_MODE = {
    "filters": {
        "episodes": 180,
        "warmup_episodes": 30,
        "updates_per_episode": 10,
        "lr_actor": 5e-7,
        "lr_actor_profile": 0.0,
        "actor_lr_final_scale": 1.0,
        "tau": 0.003,
        "initial_action": 0.18,
        "actor_t_skip": 0.0,
        "actor_x_skip": 0.0,
        "actor_z_skip": 2.0,
        "eval_episodes": 120,
        "actor_z_slope_skip": 2.0,
        "actor_z_curve_skip": 0.0,
        "actor_bias_shift": -0.30,
        "actor_update_every": 2,
        "seed": 29,
    }
}


def _run_mode(
    mode: str,
    output_dir: Path,
    python_executable: str,
    training: dict,
    warm_start_current: Path | None = None,
) -> tuple[str, str]:
    key = f"ct_{mode}"
    command = [python_executable, str(HERE / "tune_ct_powerlaw_candidate.py"), "--observation", mode]
    option_names = {
        "episodes": "--episodes",
        "warmup_episodes": "--warmup-episodes",
        "updates_per_episode": "--updates-per-episode",
        "eval_every": "--eval-every",
        "eval_episodes": "--eval-episodes",
        "checkpoint_seed": "--checkpoint-seed",
        "early_stop_patience": "--early-stop-patience",
        "validation_episodes": "--validation-episodes",
        "validation_seed": "--validation-seed",
        "history_len": "--history-len",
        "seed": "--seed",
        "blas_threads": "--blas-threads",
        "lr_actor": "--lr-actor",
        "lr_actor_profile": "--lr-actor-profile",
        "actor_lr_final_scale": "--actor-lr-final-scale",
        "lr_critic": "--lr-critic",
        "tau": "--tau",
        "initial_action": "--initial-action",
        "actor_t_skip": "--actor-t-skip",
        "actor_x_skip": "--actor-x-skip",
        "actor_z_skip": "--actor-z-skip",
        "actor_z_slope_skip": "--actor-z-slope-skip",
        "actor_z_curve_skip": "--actor-z-curve-skip",
        "actor_bias_shift": "--actor-bias-shift",
        "actor_update_every": "--actor-update-every",
    }
    for key_name, option in option_names.items():
        command.extend((option, str(training[key_name])))
    if warm_start_current is not None:
        command.extend(("--warm-start-current", str(warm_start_current)))
    command.extend(("--name", key, "--out-dir", str(output_dir)))
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    completed = subprocess.run(command, cwd=HERE, env=env, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{key} failed\n{completed.stdout}\n{completed.stderr}")
    return key, completed.stdout


def train_tuned_ct_powerlaw_suite(
    output_dir: str | Path,
    force: bool = False,
    max_workers: int = 3,
    python_executable: str | None = None,
    training: dict | None = None,
    training_by_mode: dict[str, dict] | None = None,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = {**DEFAULT_TRAINING, **(training or {})}
    per_mode = {
        mode: {**DEFAULT_TRAINING_BY_MODE.get(mode, {}), **(training_by_mode or {}).get(mode, {})}
        for mode in MODES
    }
    pending = [
        mode for mode in MODES
        if force or not (output / f"ct_{mode}.pkl").exists() or not (output / f"ct_{mode}.json").exists()
    ]
    if not pending:
        print("All tuned power-law CT-DDPG checkpoints are present; set force=True to retrain.")
        return {}
    executable = python_executable or sys.executable
    print(f"Training {len(pending)} power-law CT-DDPG jobs sequentially (current, then filtered warm start).")
    logs: dict[str, str] = {}
    if "current" in pending:
        key, log = _run_mode("current", output, executable, {**selected, **per_mode["current"]})
        logs[key] = log
        print(f"\n[{key}] complete\n" + "\n".join(log.rstrip().splitlines()[-8:]))
    if "filters" in pending:
        current_checkpoint = output / "ct_current.pkl"
        if not current_checkpoint.exists():
            raise FileNotFoundError("filtered training requires the current CT-DDPG checkpoint")
        key, log = _run_mode(
            "filters",
            output,
            executable,
            {**selected, **per_mode["filters"]},
            warm_start_current=current_checkpoint,
        )
        logs[key] = log
        print(f"\n[{key}] complete\n" + "\n".join(log.rstrip().splitlines()[-8:]))
    return logs


def load_tuned_ct_powerlaw_suite(output_dir: str | Path):
    output = Path(output_dir)
    agents = {}
    histories = {}
    metadata = {}
    for mode in MODES:
        with (output / f"ct_{mode}.pkl").open("rb") as stream:
            agents[mode] = pickle.load(stream)
        metadata[mode] = json.loads((output / f"ct_{mode}.json").read_text(encoding="utf-8"))
        histories[mode] = metadata[mode]["training_history"]
    return agents, histories, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train tuned power-law CT-DDPG variants.")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    train_tuned_ct_powerlaw_suite(args.out_dir, force=args.force, max_workers=args.max_workers)
