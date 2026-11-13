from __future__ import annotations

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
    "--env-preset",
    "memory_strong",
    "--n-envs",
    "4",
    "--hidden-dim",
    "128",
    "--learning-starts",
    "5000",
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

SAC_COMMON = (
    "--algorithm",
    "sac",
    "--batch-size",
    "512",
    "--gradient-steps",
    "1",
    "--reward-scale",
    "25",
)

DDPG_COMMON = (
    "--algorithm",
    "ddpg",
    "--feature-mode",
    "matched",
    "--timesteps",
    "160000",
    "--learning-rate",
    "0.0001",
    "--batch-size",
    "512",
    "--gradient-steps",
    "1",
    "--tau",
    "0.002",
    "--noise",
    "0.05",
    "--weight-decay",
    "0.0001",
    "--eval-freq",
    "20000",
    "--eval-episodes",
    "64",
)

TUNED_JOBS = (
    TunedJob(
        "sac_current",
        "sac",
        "current",
        "matched",
        SAC_COMMON
        + (
            "--observation",
            "current",
            "--feature-mode",
            "matched",
            "--timesteps",
            "180000",
            "--learning-rate",
            "0.0001",
            "--tau",
            "0.005",
            "--ent-coef",
            "auto_0.005",
            "--weight-decay",
            "0.00001",
            "--eval-freq",
            "10000",
            "--eval-episodes",
            "96",
            "--early-stop-patience",
            "0",
        ),
    ),
    TunedJob(
        "sac_filters",
        "sac",
        "filters",
        "lifted",
        SAC_COMMON
        + (
            "--observation",
            "filters",
            "--feature-mode",
            "lifted",
            "--timesteps",
            "220000",
            "--learning-rate",
            "0.0001",
            "--tau",
            "0.005",
            "--ent-coef",
            "auto_0.005",
            "--weight-decay",
            "0.00001",
            "--eval-freq",
            "10000",
            "--eval-episodes",
            "96",
            "--early-stop-patience",
            "0",
        ),
        warm_start_key="sac_current",
    ),
    *(
        TunedJob(
            f"ddpg_{mode}",
            "ddpg",
            mode,
            "matched",
            DDPG_COMMON + ("--observation", mode),
        )
        if mode == "current"
        else TunedJob(
            "ddpg_filters",
            "ddpg",
            "filters",
            "matched",
            DDPG_COMMON + ("--observation", "filters"),
            warm_start_key="ddpg_current",
        )
        for mode in ("current", "filters")
    ),
)


def make_tuned_jobs(configs: dict[str, dict]) -> tuple[TunedJob, ...]:
    """Build suite jobs from the notebook's editable parameter dictionary."""

    jobs = []
    for key, cfg in configs.items():
        algorithm = str(cfg["algorithm"])
        arguments = [
            "--algorithm",
            algorithm,
            "--observation",
            str(cfg["observation"]),
            "--feature-mode",
            str(cfg["feature_mode"]),
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
        else:
            raise ValueError(f"unsupported tuned algorithm {algorithm!r}")
        jobs.append(
            TunedJob(
                key=key,
                algorithm=algorithm,
                observation=str(cfg["observation"]),
                feature_mode=str(cfg["feature_mode"]),
                arguments=tuple(arguments),
                warm_start_key=cfg.get("warm_start_key"),
            )
        )
    return tuple(jobs)


def _run_job(job: TunedJob, output_dir: Path, python_executable: str) -> tuple[str, str]:
    command = [
        python_executable,
        str(HERE / "tune_sb3_candidate.py"),
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
    output_path: Path,
    executable: str,
    max_workers: int,
) -> dict[str, str]:
    logs: dict[str, str] = {}
    if not jobs:
        return logs
    worker_count = max(1, min(int(max_workers), 4, len(jobs)))
    print(f"Training {len(jobs)} SB3 jobs with {worker_count} CPU workers (2 Torch threads each).")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_run_job, job, output_path, executable): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            key, log = future.result()
            logs[key] = log
            last_lines = "\n".join(log.rstrip().splitlines()[-8:])
            print(f"\n[{key}] complete\n{last_lines}")
    return logs


def train_tuned_suite(
    output_dir: str | Path,
    force: bool = False,
    max_workers: int = 4,
    python_executable: str | None = None,
    jobs: tuple[TunedJob, ...] | None = None,
) -> dict[str, str]:
    """Train missing SAC/DDPG observation variants in bounded parallel workers."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    executable = python_executable or sys.executable
    selected_jobs = jobs or TUNED_JOBS
    pending = [job for job in selected_jobs if not _is_cached(job, output_path, force)]
    if not pending:
        print("All tuned SB3 checkpoints are already present; set force=True to retrain.")
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


def load_tuned_suite(
    output_dir: str | Path,
    jobs: tuple[TunedJob, ...] | None = None,
) -> dict[str, tuple[object, str, int, str]]:
    from stable_baselines3 import DDPG, SAC

    output_path = Path(output_dir)
    loaded = {}
    classes = {"sac": SAC, "ddpg": DDPG}
    for job in jobs or TUNED_JOBS:
        model_path = output_path / job.model_name
        if not model_path.exists():
            raise FileNotFoundError(f"missing tuned checkpoint: {model_path}")
        model = classes[job.algorithm].load(model_path, device="cpu")
        loaded[job.key] = (model, job.observation, 24, job.feature_mode)
    return loaded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the tuned Erlang SAC/DDPG suite.")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    train_tuned_suite(args.out_dir, force=args.force, max_workers=args.max_workers)
