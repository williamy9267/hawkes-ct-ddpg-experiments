from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict, replace
from pathlib import Path

from ct_ddpg_erlang import AgentConfig, evaluate_agent, train_erlang_ct_ddpg
from erlang_env import ErlangHawkesConfig


HERE = Path(__file__).resolve().parent


def strong_memory_config() -> ErlangHawkesConfig:
    return ErlangHawkesConfig()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Erlang CT-DDPG observation candidate.")
    parser.add_argument("--observation", choices=["current", "history", "filters"], required=True)
    parser.add_argument("--feature-mode", choices=["matched", "lifted"], default="matched")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--warmup-episodes", type=int, default=40)
    parser.add_argument("--updates-per-episode", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=80)
    parser.add_argument("--checkpoint-seed", type=int, default=4019)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--validation-episodes", type=int, default=400)
    parser.add_argument("--validation-seed", type=int, default=999)
    parser.add_argument("--history-len", type=int, default=24)
    parser.add_argument("--lr-actor", type=float, default=1e-5)
    parser.add_argument("--lr-actor-profile", type=float, default=0.0)
    parser.add_argument("--lr-critic", type=float, default=6e-4)
    parser.add_argument("--initial-action", type=float, default=0.32)
    parser.add_argument("--actor-t-skip", type=float, default=0.0)
    parser.add_argument("--actor-x-skip", type=float, default=1.5)
    parser.add_argument("--actor-z-skip", type=float, default=0.35)
    parser.add_argument("--actor-z-slope-skip", type=float, default=0.0)
    parser.add_argument("--actor-z-curve-skip", type=float, default=0.0)
    parser.add_argument("--actor-profile-mode", choices=["quadratic", "full"], default="quadratic")
    parser.add_argument("--actor-update-every", type=int, default=1)
    parser.add_argument("--warm-start-current", type=Path, default=None)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--blas-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", type=Path, default=HERE / "tuning_results")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = strong_memory_config()
    agent_cfg = AgentConfig(
        observation_mode=args.observation,
        feature_mode=args.feature_mode,
        history_len=args.history_len,
        lr_actor=args.lr_actor,
        lr_actor_profile=args.lr_actor_profile,
        lr_critic=args.lr_critic,
        tau=args.tau,
        initial_action=args.initial_action,
        actor_t_skip=args.actor_t_skip,
        actor_x_skip=args.actor_x_skip,
        actor_z_skip=args.actor_z_skip,
        actor_z_slope_skip=args.actor_z_slope_skip,
        actor_z_curve_skip=args.actor_z_curve_skip,
        actor_profile_mode=args.actor_profile_mode,
        actor_update_every=args.actor_update_every,
    )
    initial_current_agent = None
    if args.warm_start_current is not None:
        with args.warm_start_current.open("rb") as stream:
            initial_current_agent = pickle.load(stream)
    started = time.perf_counter()
    try:
        from threadpoolctl import threadpool_limits

        limits = threadpool_limits(limits=max(1, args.blas_threads))
    except ImportError:
        limits = None
    if limits is None:
        agent, _, history, baselines = train_erlang_ct_ddpg(
            seed=args.seed,
            episodes=args.episodes,
            warmup_episodes=args.warmup_episodes,
            updates_per_episode=args.updates_per_episode,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            eval_seed=args.checkpoint_seed,
            early_stop_patience=args.early_stop_patience,
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            initial_current_agent=initial_current_agent,
        )
    else:
        with limits:
            agent, _, history, baselines = train_erlang_ct_ddpg(
                seed=args.seed,
                episodes=args.episodes,
                warmup_episodes=args.warmup_episodes,
                updates_per_episode=args.updates_per_episode,
                eval_every=args.eval_every,
                eval_episodes=args.eval_episodes,
                eval_seed=args.checkpoint_seed,
                early_stop_patience=args.early_stop_patience,
                env_cfg=env_cfg,
                agent_cfg=agent_cfg,
                initial_current_agent=initial_current_agent,
            )
    elapsed = time.perf_counter() - started
    stats = evaluate_agent(
        env_cfg,
        agent,
        seed=args.validation_seed,
        episodes=args.validation_episodes,
    )
    payload = {
        "name": args.name,
        "observation_mode": args.observation,
        "elapsed_seconds": elapsed,
        "environment": asdict(env_cfg),
        "agent_config": asdict(agent_cfg),
        "training_history": history,
        "training_baselines": baselines,
        "evaluation": stats,
    }
    with (args.out_dir / f"{args.name}.pkl").open("wb") as stream:
        pickle.dump(agent, stream)
    with (args.out_dir / f"{args.name}.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
