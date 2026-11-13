from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TunedJob:
    key: str
    algorithm: str
    observation: str
    feature_mode: str
    arguments: tuple[str, ...]
    warm_start_key: str | None = None

    @property
    def model_name(self) -> str:
        return f"{self.key}_model.zip"

    @property
    def metadata_name(self) -> str:
        return f"{self.key}.json"


COMMON = (
    "--n-envs",
    "4",
    "--hidden-dim",
    "128",
    "--eval-envs",
    "8",
    "--validation-episodes",
    "400",
    "--validation-seed",
    "8100",
    "--seed",
    "77",
    "--threads",
    "2",
)


DEFAULT_CONFIGS = {
    "sac_current": {
        "algorithm": "sac",
        "observation": "current",
        "feature_mode": "matched",
        "timesteps": 100_000,
        "learning_rate": 2e-4,
        "batch_size": 512,
        "gradient_steps": 1,
        "tau": 0.005,
        "ent_coef": "auto_0.02",
        "weight_decay": 1e-5,
        "eval_freq": 12_500,
        "eval_episodes": 64,
    },
    "ddpg_current": {
        "algorithm": "ddpg",
        "observation": "current",
        "feature_mode": "matched",
        "timesteps": 120_000,
        "learning_rate": 5e-5,
        "batch_size": 1024,
        "gradient_steps": 1,
        "tau": 0.003,
        "noise": 0.05,
        "weight_decay": 1e-4,
        "eval_freq": 10_000,
        "eval_episodes": 128,
        "early_stop_patience": 0,
        "reward_scale": 10.0,
        "learning_starts": 15_000,
        "initial_action": 0.30,
    },
    "sac_filters": {
        "algorithm": "sac",
        "observation": "filters",
        "feature_mode": "matched",
        "timesteps": 200_000,
        "learning_rate": 1e-4,
        "batch_size": 512,
        "gradient_steps": 2,
        "n_steps": 1,
        "tau": 0.005,
        "ent_coef": "auto_0.005",
        "weight_decay": 1e-5,
        "eval_freq": 10_000,
        "eval_episodes": 96,
        "early_stop_patience": 0,
        "warm_start_key": "sac_current",
    },
    "ddpg_filters": {
        "algorithm": "ddpg",
        "observation": "filters",
        "feature_mode": "matched",
        "timesteps": 160_000,
        "learning_rate": 5e-5,
        "batch_size": 1024,
        "gradient_steps": 1,
        "tau": 0.003,
        "noise": 0.03,
        "weight_decay": 1e-4,
        "eval_freq": 10_000,
        "eval_episodes": 128,
        "early_stop_patience": 0,
        "reward_scale": 10.0,
        "learning_starts": 15_000,
        "warm_start_key": "ddpg_current",
    },
}


def make_tuned_jobs(configs: dict[str, dict]) -> tuple[TunedJob, ...]:
    jobs = []
    for key, cfg in configs.items():
        algorithm = str(cfg["algorithm"])
        arguments = [
            "--algorithm",
            algorithm,
            "--observation",
            str(cfg["observation"]),
            "--feature-mode",
            str(cfg.get("feature_mode", "matched")),
            "--timesteps",
            str(cfg["timesteps"]),
            "--learning-rate",
            str(cfg["learning_rate"]),
            "--batch-size",
            str(cfg.get("batch_size", 512)),
            "--gradient-steps",
            str(cfg.get("gradient_steps", 1)),
            "--n-steps",
            str(cfg.get("n_steps", 1)),
            "--tau",
            str(cfg["tau"]),
            "--weight-decay",
            str(cfg.get("weight_decay", 0.0)),
            "--eval-freq",
            str(cfg["eval_freq"]),
            "--eval-episodes",
            str(cfg.get("eval_episodes", 64)),
            "--early-stop-patience",
            str(cfg.get("early_stop_patience", 3)),
            "--early-stop-min-evals",
            str(cfg.get("early_stop_min_evals", 1)),
            "--reward-scale",
            str(cfg.get("reward_scale", 25.0)),
        ]
        if algorithm == "sac":
            arguments.extend(("--ent-coef", str(cfg["ent_coef"])))
        elif algorithm == "ddpg":
            arguments.extend(("--noise", str(cfg["noise"])))
            if cfg.get("initial_action") is not None:
                arguments.extend(("--ddpg-initial-action", str(cfg["initial_action"])))
        else:
            raise ValueError(f"unsupported tuned algorithm {algorithm!r}")
        if cfg.get("learning_starts") is not None:
            arguments.extend(("--learning-starts", str(cfg["learning_starts"])))
        jobs.append(
            TunedJob(
                key=key,
                algorithm=algorithm,
                observation=str(cfg["observation"]),
                feature_mode=str(cfg.get("feature_mode", "matched")),
                arguments=tuple(arguments),
                warm_start_key=cfg.get("warm_start_key"),
            )
        )
    return tuple(jobs)


TUNED_JOBS = make_tuned_jobs(DEFAULT_CONFIGS)


def _run_job(job: TunedJob, output_dir: Path, python_executable: str) -> tuple[str, str]:
    command = [
        python_executable,
        str(HERE / "tune_sb3_powerlaw_candidate.py"),
        *COMMON,
        *job.arguments,
    ]
    if job.warm_start_key is not None:
        command.extend(("--warm-start", str(output_dir / f"{job.warm_start_key}_model.zip")))
    command.extend(("--name", job.key, "--out-dir", str(output_dir)))
    process_env = os.environ.copy()
    process_env["OMP_NUM_THREADS"] = "2"
    process_env["MKL_NUM_THREADS"] = "2"
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
            f"{job.key} failed with exit code {completed.returncode}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return job.key, completed.stdout


def _is_cached(job: TunedJob, output_dir: Path, force: bool) -> bool:
    return (
        not force
        and (output_dir / job.model_name).exists()
        and (output_dir / job.metadata_name).exists()
    )


def _run_stage(
    jobs: list[TunedJob],
    output_dir: Path,
    executable: str,
    max_workers: int,
) -> dict[str, str]:
    logs: dict[str, str] = {}
    if not jobs:
        return logs
    worker_count = max(1, min(int(max_workers), 4, len(jobs)))
    print(f"Training {len(jobs)} SB3 jobs with {worker_count} CPU workers (2 Torch threads each).")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_run_job, job, output_dir, executable): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            key, log = future.result()
            logs[key] = log
            print(f"\n[{key}] complete\n" + "\n".join(log.rstrip().splitlines()[-8:]))
    return logs


def train_tuned_powerlaw_suite(
    output_dir: str | Path,
    force: bool = False,
    max_workers: int = 4,
    python_executable: str | None = None,
    jobs: tuple[TunedJob, ...] | None = None,
) -> dict[str, str]:
    """Train independent jobs first, then warm-started filtered jobs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    executable = python_executable or sys.executable
    selected = jobs or TUNED_JOBS
    pending = [job for job in selected if not _is_cached(job, output_path, force)]
    if not pending:
        print("All tuned power-law SB3 checkpoints are cached; set force=True to retrain.")
        return {}

    independent = [job for job in pending if job.warm_start_key is None]
    dependent = [job for job in pending if job.warm_start_key is not None]
    logs = _run_stage(independent, output_path, executable, max_workers)
    for job in dependent:
        dependency = output_path / f"{job.warm_start_key}_model.zip"
        if not dependency.exists():
            raise FileNotFoundError(f"missing warm-start checkpoint: {dependency}")
    logs.update(_run_stage(dependent, output_path, executable, max_workers))
    return logs


def load_tuned_powerlaw_suite(
    output_dir: str | Path,
    jobs: tuple[TunedJob, ...] | None = None,
) -> dict[str, tuple[object, str, int, str]]:
    from stable_baselines3 import DDPG, SAC

    output_path = Path(output_dir)
    classes = {"sac": SAC, "ddpg": DDPG}
    loaded = {}
    for job in jobs or TUNED_JOBS:
        model_path = output_path / job.model_name
        if not model_path.exists():
            raise FileNotFoundError(f"missing tuned checkpoint: {model_path}")
        model = classes[job.algorithm].load(model_path, device="cpu")
        loaded[job.key] = (model, job.observation, 24, job.feature_mode)
    return loaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the tuned power-law SAC/DDPG suite.")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    train_tuned_powerlaw_suite(args.out_dir, force=args.force, max_workers=args.max_workers)
