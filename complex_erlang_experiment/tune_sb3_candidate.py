from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from erlang_env import ErlangHawkesConfig
from sb3_baselines import SB3Config, evaluate_sb3_models, train_sb3_baselines


HERE = Path(__file__).resolve().parent


def parse_entropy_coefficient(value: str) -> str | float:
    return value if value.startswith("auto") else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate one Erlang SB3 tuning candidate.")
    parser.add_argument("--algorithm", choices=["sac", "ddpg"], required=True)
    parser.add_argument("--env-preset", choices=["base", "memory", "memory_strong"], default="base")
    parser.add_argument("--observation", choices=["current", "history", "filters", "exact"], default="filters")
    parser.add_argument(
        "--feature-mode",
        choices=["orthogonal", "whitened", "summary", "pca", "matched", "conditioned", "lifted", "legacy"],
        default="matched",
    )
    parser.add_argument("--timesteps", type=int, default=80_000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gradient-steps", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--ent-coef", type=str, default="auto_0.02")
    parser.add_argument("--noise", type=float, default=0.10)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--reward-scale", type=float, default=25.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--actor-only-warm-start", action="store_true")
    parser.add_argument("--warm-start-memory-noise", type=float, default=0.0)
    parser.add_argument("--history-len", type=int, default=24)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=48)
    parser.add_argument("--eval-envs", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-evals", type=int, default=1)
    parser.add_argument("--validation-episodes", type=int, default=200)
    parser.add_argument("--validation-seed", type=int, default=8100)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warm-start", type=Path, default=None)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, default=HERE / "tuning_results")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SB3Config(
        total_timesteps=args.timesteps,
        history_len=args.history_len,
        feature_mode=args.feature_mode,
        n_envs=args.n_envs,
        ddpg_n_envs=args.n_envs,
        sac_learning_rate=args.learning_rate,
        ddpg_learning_rate=args.learning_rate,
        sac_batch_size=args.batch_size,
        ddpg_batch_size=args.batch_size,
        sac_gradient_steps=args.gradient_steps,
        sac_n_steps=args.n_steps,
        ddpg_gradient_steps=args.gradient_steps,
        sac_tau=args.tau,
        ddpg_tau=args.tau,
        sac_ent_coef=parse_entropy_coefficient(args.ent_coef),
        learning_starts=args.learning_starts,
        ddpg_action_noise_sigma=args.noise,
        policy_hidden=(args.hidden_dim, args.hidden_dim),
        reward_scale=args.reward_scale,
        optimizer_weight_decay=args.weight_decay,
        warm_start_critic=not args.actor_only_warm_start,
        warm_start_memory_noise=args.warm_start_memory_noise,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        eval_n_envs=args.eval_envs,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_evals=args.early_stop_min_evals,
        restore_best=True,
        seed=args.seed,
        device="cpu",
        torch_num_threads=args.threads,
    )
    env_cfg = ErlangHawkesConfig()
    if args.env_preset == "memory":
        env_cfg = replace(
            env_cfg,
            excitation=1.00,
            control_effect_strength=1.30,
            jump_size=0.08,
            jump_size_x_strength=0.005,
        )
    elif args.env_preset == "memory_strong":
        env_cfg = ErlangHawkesConfig()
    started = time.perf_counter()
    warm_start_models = None
    if args.warm_start is not None:
        if args.algorithm == "ddpg":
            from stable_baselines3 import DDPG

            warm_start_models = {"ddpg": DDPG.load(args.warm_start, device="cpu")}
        elif args.algorithm == "sac":
            from stable_baselines3 import SAC

            warm_start_models = {"sac": SAC.load(args.warm_start, device="cpu")}
        else:
            raise ValueError("--warm-start is supported for SAC and DDPG")
    model = train_sb3_baselines(
        env_cfg,
        observation_mode=args.observation,
        cfg=cfg,
        algorithms=(args.algorithm,),
        warm_start_models=warm_start_models,
    )[args.algorithm]
    elapsed = time.perf_counter() - started
    stats = evaluate_sb3_models(
        env_cfg,
        {args.algorithm: model},
        observation_mode=args.observation,
        history_len=args.history_len,
        seed=args.validation_seed,
        episodes=args.validation_episodes,
        feature_mode=args.feature_mode,
    )[args.algorithm]

    model.save(str(args.out_dir / f"{args.name}_model"))
    payload = {
        "name": args.name,
        "algorithm": args.algorithm,
        "observation_mode": args.observation,
        "environment": asdict(env_cfg),
        "elapsed_seconds": elapsed,
        "config": asdict(cfg),
        "training_history": getattr(model, "erlang_evaluation_history", []),
        "evaluation": stats,
    }
    with (args.out_dir / f"{args.name}.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
