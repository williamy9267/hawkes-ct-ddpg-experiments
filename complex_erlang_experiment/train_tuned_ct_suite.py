from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODES = ("current", "filters")


DEFAULT_TRAINING = {
    "episodes": 140,
    "warmup_episodes": 30,
    "updates_per_episode": 10,
    "eval_every": 20,
    "eval_episodes": 100,
    "checkpoint_seed": 4019,
    "early_stop_patience": 5,
    "validation_episodes": 240,
    "history_len": 24,
    "seed": 19,
    "blas_threads": 2,
    "feature_mode": "matched",
    "lr_actor": 2e-5,
    "lr_actor_profile": 0.0,
    "lr_critic": 6e-4,
    "initial_action": 0.22,
    "actor_t_skip": 0.0,
    "actor_x_skip": 1.2,
    "actor_z_skip": 0.0,
    "actor_z_slope_skip": 0.0,
    "actor_z_curve_skip": 0.0,
    "actor_update_every": 1,
    "tau": 0.003,
    "validation_seed": 8100,
}

DEFAULT_TRAINING_BY_MODE = {
    "filters": {
        "episodes": 140,
        "warmup_episodes": 30,
        "updates_per_episode": 10,
        "eval_episodes": 100,
        "early_stop_patience": 5,
        "lr_actor": 5e-6,
        "lr_actor_profile": 1e-4,
        "initial_action": 0.16,
        "actor_t_skip": 0.0,
        "actor_x_skip": 6.0,
        "actor_z_skip": 1.0,
        "actor_z_slope_skip": 0.0,
        "actor_z_curve_skip": 0.0,
        "actor_update_every": 1,
        "tau": 0.003,
    }
}


def _run_mode(
    mode: str,
    output_dir: Path,
    python_executable: str,
    training: dict,
) -> tuple[str, str]:
    key = f"ct_{mode}"
    command = [
        python_executable,
        str(HERE / "tune_ct_candidate.py"),
        "--observation",
        mode,
        "--feature-mode",
        str(training["feature_mode"]),
        "--episodes",
        str(training["episodes"]),
        "--warmup-episodes",
        str(training["warmup_episodes"]),
        "--updates-per-episode",
        str(training["updates_per_episode"]),
        "--eval-every",
        str(training["eval_every"]),
        "--eval-episodes",
        str(training["eval_episodes"]),
        "--checkpoint-seed",
        str(training["checkpoint_seed"]),
        "--early-stop-patience",
        str(training["early_stop_patience"]),
        "--validation-episodes",
        str(training["validation_episodes"]),
        "--history-len",
        str(training["history_len"]),
        "--seed",
        str(training["seed"]),
        "--blas-threads",
        str(training["blas_threads"]),
        "--lr-actor",
        str(training["lr_actor"]),
        "--lr-actor-profile",
        str(training["lr_actor_profile"]),
        "--lr-critic",
        str(training["lr_critic"]),
        "--initial-action",
        str(training["initial_action"]),
        "--actor-t-skip",
        str(training["actor_t_skip"]),
        "--actor-x-skip",
        str(training["actor_x_skip"]),
        "--actor-z-skip",
        str(training["actor_z_skip"]),
        "--actor-z-slope-skip",
        str(training["actor_z_slope_skip"]),
        "--actor-z-curve-skip",
        str(training["actor_z_curve_skip"]),
        "--actor-update-every",
        str(training["actor_update_every"]),
        "--tau",
        str(training["tau"]),
        "--validation-seed",
        str(training["validation_seed"]),
        "--name",
        key,
        "--out-dir",
        str(output_dir),
    ]
    process_env = os.environ.copy()
    process_env["OMP_NUM_THREADS"] = "2"
    process_env["MKL_NUM_THREADS"] = "2"
    process_env["OPENBLAS_NUM_THREADS"] = "2"
    completed = subprocess.run(
        command,
        cwd=HERE,
        env=process_env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{key} failed with exit code {completed.returncode}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return key, completed.stdout


def train_tuned_ct_suite(
    output_dir: str | Path,
    force: bool = False,
    max_workers: int = 3,
    python_executable: str | None = None,
    training: dict | None = None,
    training_by_mode: dict[str, dict] | None = None,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    executable = python_executable or sys.executable
    selected_training = {**DEFAULT_TRAINING, **(training or {})}
    per_mode = {
        mode: {**DEFAULT_TRAINING_BY_MODE.get(mode, {}), **(training_by_mode or {}).get(mode, {})}
        for mode in MODES
    }
    pending = [
        mode
        for mode in MODES
        if force
        or not (output_path / f"ct_{mode}.pkl").exists()
        or not (output_path / f"ct_{mode}.json").exists()
    ]
    if not pending:
        print("All tuned CT-DDPG checkpoints are already present; set force=True to retrain.")
        return {}

    worker_count = max(1, min(int(max_workers), 3, len(pending)))
    print(f"Training {len(pending)} CT-DDPG jobs with {worker_count} CPU workers.")
    logs: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _run_mode,
                mode,
                output_path,
                executable,
                {**selected_training, **per_mode.get(mode, {})},
            ): mode
            for mode in pending
        }
        for future in as_completed(futures):
            key, log = future.result()
            logs[key] = log
            print(f"\n[{key}] complete\n" + "\n".join(log.rstrip().splitlines()[-8:]))
    return logs


def load_tuned_ct_suite(output_dir: str | Path):
    output_path = Path(output_dir)
    agents = {}
    histories = {}
    metadata = {}
    for mode in MODES:
        pickle_path = output_path / f"ct_{mode}.pkl"
        json_path = output_path / f"ct_{mode}.json"
        if not pickle_path.exists() or not json_path.exists():
            raise FileNotFoundError(f"missing tuned CT-DDPG checkpoint for {mode}: {output_path}")
        with pickle_path.open("rb") as stream:
            agents[mode] = pickle.load(stream)
        metadata[mode] = json.loads(json_path.read_text(encoding="utf-8"))
        histories[mode] = metadata[mode]["training_history"]
    return agents, histories, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the tuned Erlang CT-DDPG suite.")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    train_tuned_ct_suite(args.out_dir, force=args.force, max_workers=args.max_workers)
