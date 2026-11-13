from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PROJECT_HELPER_ROOT = next(
    (
        parent
        for parent in (HERE, *HERE.parents)
        if (parent / "neural_numpy.py").exists() and (parent / "replay.py").exists()
    ),
    ROOT,
)
for path in (PROJECT_HELPER_ROOT, ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from neural_numpy import Actor, Adam, MLP, grad_norm
from replay import Episode, ReplayBuffer

from current_readout_env import CurrentReadoutEnvConfig, SingleExpCurrentReadoutEnv, state_features


@dataclass
class AgentConfig:
    hidden: Tuple[int, int] = (48, 48)
    segment_len: int = 50
    batch_size: int = 128
    actor_batch_size: int = 256
    lr_actor: float = 2e-5
    lr_critic: float = 8e-4
    tau: float = 0.01
    terminal_penalty: float = 5.0
    grad_clip: float = 10.0
    initial_action: float = 0.30
    actor_x_skip: float = 6.0
    actor_z_skip: float = 0.5


class CTDDPGCurrentReadoutAgent:
    """CT-DDPG-style actor-critic on the Section 1 Markov state (t, X, Z)."""

    def __init__(self, env_cfg: CurrentReadoutEnvConfig, cfg: AgentConfig, rng: np.random.Generator):
        self.env_cfg = env_cfg
        self.cfg = cfg
        hidden = list(cfg.hidden)
        feature_dim = env_cfg.feature_dim
        actor_skip = np.zeros(feature_dim, dtype=np.float64)
        actor_skip[1] = cfg.actor_x_skip
        actor_skip[2:] = cfg.actor_z_skip
        self.actor = Actor(
            feature_dim,
            hidden,
            rng,
            initial_action=cfg.initial_action,
            fixed_input_skip=actor_skip,
        )
        self.value = MLP([feature_dim, *hidden, 1], rng, output_scale=0.30)
        self.target_value = MLP([feature_dim, *hidden, 1], rng, output_scale=0.30)
        self.target_value.copy_from(self.value)
        self.qbar = MLP([feature_dim + 1, *hidden, 1], rng, output_scale=0.30)
        self.actor_opt = Adam(self.actor.net, cfg.lr_actor, cfg.grad_clip)
        self.value_opt = Adam(self.value, cfg.lr_critic, cfg.grad_clip)
        self.qbar_opt = Adam(self.qbar, cfg.lr_critic, cfg.grad_clip)

    def features(self, times: np.ndarray, states: np.ndarray) -> np.ndarray:
        return state_features(times, states, self.env_cfg)

    def act(self, time: float, state: np.ndarray) -> float:
        feats = self.features(np.asarray([time]), np.asarray([state]))
        return float(self.actor.forward(feats)[0, 0])

    def value_path(self, times: np.ndarray, states: np.ndarray) -> np.ndarray:
        feats = self.features(times, states)
        return self.value.forward(feats).reshape(-1)

    def critic_update(self, buffer: ReplayBuffer, rng: np.random.Generator) -> dict:
        cfg = self.cfg
        env_cfg = self.env_cfg
        times, states, actions, reward_rates, terminal = buffer.sample_segments(
            cfg.batch_size, cfg.segment_len, rng
        )
        batch = times.shape[0]
        l_steps = cfg.segment_len
        discounts = np.exp(-env_cfg.beta * env_cfg.dt * np.arange(l_steps))
        end_discount = float(np.exp(-env_cfg.beta * env_cfg.dt * l_steps))

        start_feats = self.features(times[:, 0], states[:, 0])
        end_feats = self.features(times[:, -1], states[:, -1])
        step_feats = self.features(
            times[:, :-1].reshape(-1),
            states[:, :-1].reshape(-1, env_cfg.state_dim),
        )
        flat_actions = actions.reshape(-1, 1)

        v_start, v_cache = self.value.forward(start_feats, cache=True)
        v_end = self.target_value.forward(end_feats)
        terminal_values = SingleExpCurrentReadoutEnv(env_cfg).terminal_reward(states[:, -1])[:, None]
        v_boot = np.where(terminal[:, None], terminal_values, v_end)

        policy_actions = self.actor.forward(step_feats)
        q_sa_input = np.concatenate([step_feats, flat_actions], axis=1)
        q_mu_input = np.concatenate([step_feats, policy_actions], axis=1)
        q_sa, q_sa_cache = self.qbar.forward(q_sa_input, cache=True)
        q_mu, q_mu_cache = self.qbar.forward(q_mu_input, cache=True)
        q_adv = (q_sa - q_mu).reshape(batch, l_steps)
        q_adv_flat = q_adv.reshape(-1, 1)

        td_sum = np.sum(discounts[None, :] * (reward_rates - q_adv) * env_cfg.dt, axis=1, keepdims=True)
        residual = end_discount * v_boot - v_start + td_sum
        critic_loss = float(np.mean(residual * residual))

        term_times, term_states = buffer.sample_states(cfg.batch_size, rng, terminal=True)
        term_feats = self.features(term_times, term_states)
        target_term = SingleExpCurrentReadoutEnv(env_cfg).terminal_reward(term_states)[:, None]
        v_term, v_term_cache = self.value.forward(term_feats, cache=True)
        term_error = v_term - target_term
        terminal_loss = float(np.mean(term_error * term_error))

        self.value.zero_grad()
        self.qbar.zero_grad()
        d_residual = 2.0 * residual / batch
        self.value.backward(v_cache, -d_residual, accumulate=True)

        d_q = (-d_residual * discounts[None, :] * env_cfg.dt).reshape(-1, 1)
        self.qbar.backward(q_sa_cache, d_q, accumulate=True)
        self.qbar.backward(q_mu_cache, -d_q, accumulate=True)

        d_v_term = cfg.terminal_penalty * 2.0 * term_error / cfg.batch_size
        self.value.backward(v_term_cache, d_v_term, accumulate=True)

        value_grad_norm = grad_norm(self.value)
        qbar_grad_norm = grad_norm(self.qbar)
        self.value_opt.step()
        self.qbar_opt.step()
        total_loss = critic_loss + cfg.terminal_penalty * terminal_loss
        return {
            "critic_loss": total_loss,
            "td_loss": critic_loss,
            "terminal_loss": terminal_loss,
            "mean_abs_td": float(np.mean(np.abs(residual))),
            "advantage_rate_abs": float(np.mean(np.abs(q_adv_flat))),
            "advantage_rate_rms": float(np.sqrt(np.mean(q_adv_flat * q_adv_flat))),
            "value_grad_norm": value_grad_norm,
            "qbar_grad_norm": qbar_grad_norm,
        }

    def actor_update(self, buffer: ReplayBuffer, rng: np.random.Generator) -> dict:
        cfg = self.cfg
        times, states = buffer.sample_states(cfg.actor_batch_size, rng, terminal=False)
        feats = self.features(times, states)
        actions, actor_cache = self.actor.forward(feats, cache=True)
        q_input = np.concatenate([feats, actions], axis=1)
        q_values, q_cache = self.qbar.forward(q_input, cache=True)
        grad_q_input = self.qbar.backward(
            q_cache, np.ones_like(q_values) / cfg.actor_batch_size, accumulate=False
        )
        grad_q_action = grad_q_input[:, -1:]

        self.actor.zero_grad()
        self.actor.backward(actor_cache, -grad_q_action)
        actor_grad_norm = grad_norm(self.actor.net)
        self.actor_opt.step()
        self.target_value.soft_update_from(self.value, cfg.tau)
        return {
            "actor_loss": float(-np.mean(q_values)),
            "mean_action": float(np.mean(actions)),
            "mean_qbar": float(np.mean(q_values)),
            "actor_grad_norm": actor_grad_norm,
            "action_grad_abs": float(np.mean(np.abs(grad_q_action))),
        }

    def update(self, buffer: ReplayBuffer, rng: np.random.Generator, updates: int) -> dict:
        metrics = {}
        for _ in range(updates):
            metrics.update(self.critic_update(buffer, rng))
            metrics.update(self.actor_update(buffer, rng))
        return metrics


def _copy_net_state(net: MLP) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return ([w.copy() for w in net.weights], [b.copy() for b in net.biases])


def _restore_net_state(net: MLP, state: tuple[list[np.ndarray], list[np.ndarray]]) -> None:
    weights, biases = state
    for dst, src in zip(net.weights, weights):
        dst[...] = src
    for dst, src in zip(net.biases, biases):
        dst[...] = src


def snapshot_agent(agent: CTDDPGCurrentReadoutAgent) -> dict:
    return {
        "actor": _copy_net_state(agent.actor.net),
        "value": _copy_net_state(agent.value),
        "target_value": _copy_net_state(agent.target_value),
        "qbar": _copy_net_state(agent.qbar),
    }


def restore_agent(agent: CTDDPGCurrentReadoutAgent, state: dict) -> None:
    _restore_net_state(agent.actor.net, state["actor"])
    _restore_net_state(agent.value, state["value"])
    _restore_net_state(agent.target_value, state["target_value"])
    _restore_net_state(agent.qbar, state["qbar"])


def collect_episode(
    env_cfg: CurrentReadoutEnvConfig,
    agent: CTDDPGCurrentReadoutAgent | None,
    rng: np.random.Generator,
    exploration_std: float = 0.0,
    constant_action: float | None = None,
) -> Episode:
    env = SingleExpCurrentReadoutEnv(env_cfg)
    state = env.reset()
    states = [state.copy()]
    times = [0.0]
    actions = []
    rewards = []
    jumps = []
    intensities = []
    discounted_return = 0.0

    for k in range(env_cfg.steps):
        t = k * env_cfg.dt
        if constant_action is not None:
            action = constant_action
        elif agent is None:
            action = rng.uniform(0.0, 1.0)
        else:
            action = agent.act(t, state)
            if exploration_std > 0.0:
                action += rng.normal(0.0, exploration_std)
        action = float(np.clip(action, 0.0, 1.0))
        next_state, reward_rate, done, info = env.step(action, rng)
        discounted_return += np.exp(-env_cfg.beta * t) * reward_rate * env_cfg.dt
        actions.append(action)
        rewards.append(reward_rate)
        jumps.append(info["jumps"])
        intensities.append(info["lambda"])
        state = next_state
        states.append(state.copy())
        times.append((k + 1) * env_cfg.dt)
        if done:
            break

    terminal = env.terminal_reward(np.asarray([states[-1]]))[0]
    discounted_return += np.exp(-env_cfg.beta * env_cfg.horizon) * terminal
    return Episode(
        times=np.asarray(times, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.float64),
        reward_rates=np.asarray(rewards, dtype=np.float64),
        jumps=np.asarray(jumps, dtype=np.int64),
        intensities=np.asarray(intensities, dtype=np.float64),
        discounted_return=float(discounted_return),
    )


def evaluate_agent(
    env_cfg: CurrentReadoutEnvConfig,
    agent: CTDDPGCurrentReadoutAgent | None,
    seed: int,
    episodes: int = 20,
    constant_action: float | None = None,
) -> dict:
    returns = []
    actions = []
    jumps = []
    intensities = []
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        ep = collect_episode(env_cfg, agent, rng, exploration_std=0.0, constant_action=constant_action)
        returns.append(ep.discounted_return)
        actions.append(float(np.mean(ep.actions)))
        jumps.append(float(np.sum(ep.jumps)))
        intensities.append(float(np.mean(ep.intensities)))
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_se": float(np.std(returns) / np.sqrt(max(1, episodes))),
        "return_ci90": float(1.6448536269514722 * np.std(returns) / np.sqrt(max(1, episodes))),
        "action_mean": float(np.mean(actions)),
        "jumps_mean": float(np.mean(jumps)),
        "intensity_mean": float(np.mean(intensities)),
        "episodes": int(episodes),
    }


def estimate_cost_to_go(
    env_cfg: CurrentReadoutEnvConfig,
    agent,
    times: np.ndarray,
    states: np.ndarray,
    seed: int = 1234,
    rollouts: int = 3,
    constant_action: float | None = None,
) -> np.ndarray:
    """Monte Carlo cost-to-go from arbitrary grid states.

    This is slower than reading a critic, but it puts CT-DDPG and DGM policies
    on the same cost scale and avoids comparing two different value
    parameterizations.
    """
    times_arr = np.asarray(times, dtype=np.float64).reshape(-1)
    states_arr = np.asarray(states, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.zeros(times_arr.shape[0], dtype=np.float64)
    for i, (time, state) in enumerate(zip(times_arr, states_arr)):
        rollout_costs = []
        for _ in range(rollouts):
            env = SingleExpCurrentReadoutEnv(env_cfg)
            state_now = env.set_state(time, state)
            running = 0.0
            start_k = env.k
            while env.k < env_cfg.steps:
                t_local = (env.k - start_k) * env_cfg.dt
                if constant_action is not None:
                    action = constant_action
                else:
                    action = agent.act(env.t, state_now)
                state_now, reward_rate, done, _ = env.step(float(np.clip(action, 0.0, 1.0)), rng)
                running += np.exp(-env_cfg.beta * t_local) * (-reward_rate) * env_cfg.dt
                if done:
                    break
            terminal_cost = env_cfg.terminal_x_cost * env.x * env.x
            terminal_discount = np.exp(-env_cfg.beta * (env.k - start_k) * env_cfg.dt)
            rollout_costs.append(running + terminal_discount * terminal_cost)
        values[i] = float(np.mean(rollout_costs))
    return values


def train_current_readout(
    seed: int = 11,
    episodes: int = 220,
    warmup_episodes: int = 25,
    updates_per_episode: int = 10,
    eval_every: int = 20,
    eval_episodes: int = 160,
    validation_seed: int | None = 2718,
    restore_best: bool = True,
    env_cfg: CurrentReadoutEnvConfig | None = None,
    agent_cfg: AgentConfig | None = None,
) -> Tuple[CTDDPGCurrentReadoutAgent, ReplayBuffer, List[dict], dict]:
    env_cfg = env_cfg or CurrentReadoutEnvConfig()
    agent_cfg = agent_cfg or AgentConfig()
    rng = np.random.default_rng(seed)
    agent = CTDDPGCurrentReadoutAgent(env_cfg, agent_cfg, rng)
    buffer = ReplayBuffer(max_episodes=500)

    baselines = {
        f"constant_{a:.2f}": evaluate_agent(env_cfg, None, seed + int(1000 * a) + 100, constant_action=a)
        for a in (0.0, 0.25, 0.5, 0.75, 1.0)
    }

    history: List[dict] = []
    latest_metrics: dict = {}
    best_eval_return = -np.inf
    best_state = snapshot_agent(agent)
    best_episode = 0
    for ep in range(1, episodes + 1):
        frac = min(1.0, ep / max(1, episodes))
        exploration = max(0.015, 0.20 * (1.0 - frac))
        collector = None if ep <= warmup_episodes else agent
        episode = collect_episode(env_cfg, collector, rng, exploration_std=exploration)
        buffer.add(episode)

        if len(buffer) >= warmup_episodes:
            latest_metrics = agent.update(buffer, rng, updates_per_episode)

        if ep == 1 or ep % eval_every == 0 or ep == episodes:
            eval_seed = seed + 1000 + ep if validation_seed is None else validation_seed
            eval_stats = evaluate_agent(env_cfg, agent, eval_seed, episodes=eval_episodes)
            is_best = eval_stats["return_mean"] > best_eval_return
            if is_best:
                best_eval_return = eval_stats["return_mean"]
                best_state = snapshot_agent(agent)
                best_episode = ep
            row = {
                "episode": ep,
                "train_return": episode.discounted_return,
                "exploration_std": exploration,
                "is_best_checkpoint": float(is_best),
                "best_episode": best_episode,
                "eval_seed": eval_seed,
                **latest_metrics,
                **{f"eval_{k}": v for k, v in eval_stats.items()},
            }
            history.append(row)
            print(format_log_row(row))
    if restore_best:
        restore_agent(agent, best_state)
    return agent, buffer, history, baselines


def format_log_row(row: dict) -> str:
    return (
        "ep={episode:04d} train_return={train_return:8.3f} "
        "eval_return={eval_return_mean:8.3f} action={eval_action_mean:5.3f} "
        "jumps={eval_jumps_mean:5.2f} td={mean_abs_td:7.4f} "
        "adv={advantage_rate_abs:7.4f} actor_grad={actor_grad_norm:7.4f}"
    ).format(
        **{
            "mean_abs_td": float("nan"),
            "advantage_rate_abs": float("nan"),
            "actor_grad_norm": float("nan"),
            **row,
        }
    )
