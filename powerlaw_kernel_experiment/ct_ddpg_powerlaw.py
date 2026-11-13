from __future__ import annotations

import sys
from collections import deque
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

from powerlaw_env import PowerLawHawkesConfig, PowerLawHawkesEnv, method_state_features


@dataclass
class AgentConfig:
    observation_mode: str = "filters"
    history_len: int = 24
    hidden: Tuple[int, int] = (64, 64)
    segment_len: int = 64
    batch_size: int = 192
    actor_batch_size: int = 384
    lr_actor: float = 1e-5
    actor_lr_final_scale: float = 1.0
    lr_critic: float = 6e-4
    tau: float = 0.005
    terminal_penalty: float = 7.5
    grad_clip: float = 5.0
    replay_max_episodes: int = 180
    initial_action: float = 0.32
    actor_t_skip: float = 0.0
    actor_x_skip: float = 1.5
    actor_z_skip: float = 0.35
    actor_z_slope_skip: float = 0.0
    actor_z_curve_skip: float = 0.0
    actor_bias_shift: float = 0.0
    actor_profile_mode: str = "quadratic"
    lr_actor_profile: float = 0.0
    actor_update_every: int = 1


class CTDDPGPowerLawAgent:
    """CT-DDPG actor-critic with a selectable observed-state encoder."""

    def __init__(self, env_cfg: PowerLawHawkesConfig, cfg: AgentConfig, rng: np.random.Generator):
        self.env_cfg = env_cfg
        self.cfg = cfg
        if cfg.observation_mode not in {"filters", "history", "current"}:
            raise ValueError("observation_mode must be 'filters', 'history', or 'current'")
        self.observation_mode = cfg.observation_mode
        self.history_len = int(cfg.history_len)
        hidden = list(cfg.hidden)
        if self.observation_mode == "filters":
            feature_dim = env_cfg.method_feature_dim
        elif self.observation_mode == "history":
            feature_dim = 2 + self.history_len
        else:
            feature_dim = 2
        self.feature_dim = feature_dim
        actor_skip = np.zeros(feature_dim, dtype=np.float64)
        profile_basis = None
        actor_skip[0] = cfg.actor_t_skip
        actor_skip[1] = cfg.actor_x_skip
        if self.observation_mode == "filters":
            # Keep the aggregate memory feedback invariant when the dictionary
            # size changes; otherwise K correlated filters multiply this skip by K.
            rates = np.asarray(env_cfg.filter_betas, dtype=np.float64)
            centered = (rates - rates.mean()) / max(1e-12, 0.5 * np.ptp(rates))
            curvature = centered * centered - np.mean(centered * centered)
            if cfg.actor_profile_mode == "full":
                profile_basis = np.zeros((feature_dim, feature_dim - 2), dtype=np.float64)
                profile_basis[2:, :] = np.eye(feature_dim - 2)
            elif cfg.actor_profile_mode == "quadratic":
                profile_basis = np.zeros((feature_dim, 3), dtype=np.float64)
                profile_basis[2:, 0] = 1.0 / max(1, feature_dim - 2)
                profile_basis[2:, 1] = centered / max(1, feature_dim - 2)
                profile_basis[2:, 2] = curvature / max(1, feature_dim - 2)
            else:
                raise ValueError("actor_profile_mode must be 'quadratic' or 'full'")
            actor_skip[2:] = (
                cfg.actor_z_skip
                + cfg.actor_z_slope_skip * centered
                + cfg.actor_z_curve_skip * curvature
            ) / max(1, feature_dim - 2)
        elif self.observation_mode == "history":
            actor_skip[2:] = cfg.actor_z_skip / max(1, feature_dim - 2)
        else:
            actor_skip[2:] = cfg.actor_z_skip
        self.actor = Actor(
            feature_dim,
            hidden,
            rng,
            initial_action=cfg.initial_action,
            fixed_input_skip=actor_skip,
            trainable_skip_basis=profile_basis,
        )
        self.target_actor = Actor(
            feature_dim,
            hidden,
            rng,
            initial_action=cfg.initial_action,
            fixed_input_skip=actor_skip,
            trainable_skip_basis=profile_basis,
        )
        self.target_actor.net.copy_from(self.actor.net)
        self.target_actor.copy_profile_from(self.actor)
        if cfg.actor_bias_shift != 0.0:
            self.actor.net.biases[-1][0, 0] += cfg.actor_bias_shift
            self.target_actor.net.biases[-1][0, 0] += cfg.actor_bias_shift
        self.value = MLP([feature_dim, *hidden, 1], rng, output_scale=0.30)
        self.target_value = MLP([feature_dim, *hidden, 1], rng, output_scale=0.30)
        self.target_value.copy_from(self.value)
        self.qbar = MLP([feature_dim + 1, *hidden, 1], rng, output_scale=0.30)
        self.actor_opt = Adam(self.actor.net, cfg.lr_actor, cfg.grad_clip)
        self._profile_m = np.zeros_like(self.actor.profile_coefficients)
        self._profile_v = np.zeros_like(self.actor.profile_coefficients)
        self._profile_t = 0
        self.value_opt = Adam(self.value, cfg.lr_critic, cfg.grad_clip)
        self.qbar_opt = Adam(self.qbar, cfg.lr_critic, cfg.grad_clip)
        self._update_step = 0

    def features(self, times: np.ndarray, states: np.ndarray) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float64)
        if self.observation_mode == "filters":
            return method_state_features(times, states_arr[..., : self.env_cfg.state_dim], self.env_cfg)

        times_arr = np.asarray(times, dtype=np.float64)
        t_feat = 2.0 * times_arr[..., None] / self.env_cfg.horizon - 1.0
        x_feat = np.clip(states_arr[..., 0:1] / self.env_cfg.x_scale, -5.0, 5.0)
        if self.observation_mode == "current":
            return np.concatenate([t_feat, x_feat], axis=-1)

        start = self.env_cfg.state_dim
        stop = start + self.history_len
        if states_arr.shape[-1] < stop:
            raise ValueError("history observations are missing from the replayed state")
        return np.concatenate([t_feat, x_feat, states_arr[..., start:stop]], axis=-1)

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
            states[:, :-1].reshape(-1, states.shape[-1]),
        )
        flat_actions = actions.reshape(-1, 1)

        v_start, v_cache = self.value.forward(start_feats, cache=True)
        v_end = self.target_value.forward(end_feats)
        terminal_values = PowerLawHawkesEnv(env_cfg).terminal_reward(states[:, -1])[:, None]
        v_boot = np.where(terminal[:, None], terminal_values, v_end)

        policy_actions = self.target_actor.forward(step_feats)
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
        target_term = PowerLawHawkesEnv(env_cfg).terminal_reward(term_states)[:, None]
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
        self.target_value.soft_update_from(self.value, cfg.tau)
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
        grad_q_input = self.qbar.backward(q_cache, np.ones_like(q_values) / cfg.actor_batch_size, accumulate=False)
        grad_q_action = grad_q_input[:, -1:]

        self.actor.zero_grad()
        self.actor.backward(actor_cache, -grad_q_action)
        actor_grad_norm = grad_norm(self.actor.net)
        self.actor_opt.step()
        profile_step_norm = 0.0
        if cfg.lr_actor_profile > 0.0 and self.actor.profile_coefficients.size:
            self._profile_t += 1
            profile_grad = np.clip(self.actor.profile_grad, -cfg.grad_clip, cfg.grad_clip)
            self._profile_m = 0.9 * self._profile_m + 0.1 * profile_grad
            self._profile_v = 0.999 * self._profile_v + 0.001 * (profile_grad * profile_grad)
            m_hat = self._profile_m / (1.0 - 0.9**self._profile_t)
            v_hat = self._profile_v / (1.0 - 0.999**self._profile_t)
            profile_step = cfg.lr_actor_profile * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.actor.profile_coefficients -= profile_step
            profile_step_norm = float(np.linalg.norm(profile_step))
        self.target_actor.net.soft_update_from(self.actor.net, cfg.tau)
        self.target_actor.soft_update_profile_from(self.actor, cfg.tau)
        target_actions = self.target_actor.forward(feats)
        return {
            "actor_loss": float(-np.mean(q_values)),
            "mean_action": float(np.mean(actions)),
            "mean_qbar": float(np.mean(q_values)),
            "actor_grad_norm": actor_grad_norm,
            "action_grad_abs": float(np.mean(np.abs(grad_q_action))),
            "target_actor_lag_abs": float(np.mean(np.abs(actions - target_actions))),
            "actor_profile_step_norm": profile_step_norm,
            "actor_profile_level": float(self.actor.profile_coefficients[0, 0]) if self.actor.profile_coefficients.size else 0.0,
            "actor_profile_slope": float(self.actor.profile_coefficients[1, 0]) if self.actor.profile_coefficients.size else 0.0,
            "actor_profile_curve": float(self.actor.profile_coefficients[2, 0]) if self.actor.profile_coefficients.size else 0.0,
        }

    def update(self, buffer: ReplayBuffer, rng: np.random.Generator, updates: int) -> dict:
        metrics = {}
        for _ in range(updates):
            metrics.update(self.critic_update(buffer, rng))
            self._update_step += 1
            if self._update_step % max(1, int(self.cfg.actor_update_every)) == 0:
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


def snapshot_agent(agent: CTDDPGPowerLawAgent) -> dict:
    return {
        "actor": _copy_net_state(agent.actor.net),
        "target_actor": _copy_net_state(agent.target_actor.net),
        "actor_profile": agent.actor.profile_coefficients.copy(),
        "target_actor_profile": agent.target_actor.profile_coefficients.copy(),
        "value": _copy_net_state(agent.value),
        "target_value": _copy_net_state(agent.target_value),
        "qbar": _copy_net_state(agent.qbar),
    }


def restore_agent(agent: CTDDPGPowerLawAgent, state: dict) -> None:
    _restore_net_state(agent.actor.net, state["actor"])
    if "target_actor" in state:
        _restore_net_state(agent.target_actor.net, state["target_actor"])
    else:
        agent.target_actor.net.copy_from(agent.actor.net)
    agent.actor._ensure_profile_state()
    agent.target_actor._ensure_profile_state()
    if "actor_profile" in state and agent.actor.profile_coefficients.shape == state["actor_profile"].shape:
        agent.actor.profile_coefficients[...] = state["actor_profile"]
    if "target_actor_profile" in state and agent.target_actor.profile_coefficients.shape == state["target_actor_profile"].shape:
        agent.target_actor.profile_coefficients[...] = state["target_actor_profile"]
    elif agent.target_actor.profile_coefficients.shape == agent.actor.profile_coefficients.shape:
        agent.target_actor.copy_profile_from(agent.actor)
    _restore_net_state(agent.value, state["value"])
    _restore_net_state(agent.target_value, state["target_value"])
    _restore_net_state(agent.qbar, state["qbar"])


def _lift_mlp_inputs(dst: MLP, src: MLP, action_is_last: bool = False) -> None:
    """Embed a current-state network in a larger filtered-input network."""

    if len(dst.weights) != len(src.weights) or dst.weights[0].shape[1] != src.weights[0].shape[1]:
        raise ValueError("incompatible network architectures for current-to-filter lift")
    dst.weights[0].fill(0.0)
    dst.weights[0][:2, :] = src.weights[0][:2, :]
    if action_is_last:
        dst.weights[0][-1, :] = src.weights[0][-1, :]
    for index in range(1, len(dst.weights)):
        if dst.weights[index].shape != src.weights[index].shape:
            raise ValueError("incompatible hidden layers for current-to-filter lift")
        dst.weights[index][...] = src.weights[index]
    for dst_bias, src_bias in zip(dst.biases, src.biases):
        dst_bias[...] = src_bias


def lift_current_agent_to_filtered(
    filtered_agent: CTDDPGPowerLawAgent,
    current_agent: CTDDPGPowerLawAgent,
) -> None:
    """Warm-start a filtered agent without introducing kernel information.

    The lifted networks reproduce the current-state policy/value exactly at
    initialization. Added filter coordinates and profile coefficients start at
    zero and are learned only through CT-DDPG updates.
    """

    if filtered_agent.observation_mode != "filters" or current_agent.observation_mode != "current":
        raise ValueError("the lift requires a current source and filtered destination")
    for destination, source in (
        (filtered_agent.actor.net, current_agent.actor.net),
        (filtered_agent.target_actor.net, current_agent.target_actor.net),
        (filtered_agent.value, current_agent.value),
        (filtered_agent.target_value, current_agent.target_value),
    ):
        _lift_mlp_inputs(destination, source)
    _lift_mlp_inputs(filtered_agent.qbar, current_agent.qbar, action_is_last=True)
    # Keep any configured generic filter skip while replacing the shared
    # time/state coordinates with the trained current-policy coefficients.
    filtered_agent.actor.fixed_input_skip[:2, :] = current_agent.actor.fixed_input_skip[:2, :]
    filtered_agent.target_actor.fixed_input_skip[...] = filtered_agent.actor.fixed_input_skip
    filtered_agent.actor.profile_coefficients.fill(0.0)
    filtered_agent.target_actor.profile_coefficients.fill(0.0)
    # Network lifting copies the source output bias, so reapply the filtered
    # policy's predeclared centering shift after the copy.
    if filtered_agent.cfg.actor_bias_shift != 0.0:
        filtered_agent.actor.net.biases[-1][0, 0] += filtered_agent.cfg.actor_bias_shift
        filtered_agent.target_actor.net.biases[-1][0, 0] += filtered_agent.cfg.actor_bias_shift


def collect_episode(
    env_cfg: PowerLawHawkesConfig,
    agent,
    rng: np.random.Generator,
    exploration_std: float = 0.0,
    constant_action: float | None = None,
    observation_mode: str | None = None,
    history_len: int | None = None,
) -> Episode:
    env = PowerLawHawkesEnv(env_cfg)
    state = env.reset()
    mode = observation_mode or getattr(agent, "observation_mode", "filters")
    history_size = int(history_len or getattr(agent, "history_len", 24))
    if mode not in {"filters", "history", "current"}:
        raise ValueError("observation_mode must be 'filters', 'history', or 'current'")
    event_times: deque[float] = deque(maxlen=history_size)

    def policy_state(raw_state: np.ndarray, time: float) -> np.ndarray:
        if mode != "history":
            return raw_state.copy()
        recencies = np.zeros(history_size, dtype=np.float64)
        if event_times:
            ages = np.clip(time - np.asarray(event_times), 0.0, env_cfg.horizon)
            recencies[-len(event_times) :] = np.exp(-ages / env_cfg.horizon)
        return np.concatenate([raw_state, recencies])

    states = [policy_state(state, 0.0)]
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
            action = agent.act(t, policy_state(state, t))
            if exploration_std > 0.0:
                action += rng.normal(0.0, exploration_std)
        action = float(np.clip(action, 0.0, 1.0))
        next_state, reward_rate, done, info = env.step(action, rng)
        discounted_return += np.exp(-env_cfg.beta * t) * reward_rate * env_cfg.dt
        actions.append(action)
        rewards.append(reward_rate)
        jumps.append(info["jumps"])
        intensities.append(info["lambda"])
        event_time = max(0.0, env.t - 0.5 * env_cfg.dt)
        for _ in range(int(info["jumps"])):
            event_times.append(event_time)
        state = next_state
        states.append(policy_state(state, env.t))
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


def collect_episode_from_state(
    env_cfg: PowerLawHawkesConfig,
    agent,
    rng: np.random.Generator,
    start_time: float,
    start_state: np.ndarray,
    exploration_std: float = 0.0,
    constant_action: float | None = None,
    observation_mode: str | None = None,
    history_len: int | None = None,
) -> Episode:
    """Roll out from a supplied state and discount relative to start_time.

    A history-mode rollout starts with an empty event history because past
    event times cannot be recovered from an arbitrary physical start state.
    """

    env = PowerLawHawkesEnv(env_cfg)
    state = env.set_state(start_time, np.asarray(start_state)[: env_cfg.state_dim])
    start_time = env.t
    mode = observation_mode or getattr(agent, "observation_mode", "filters")
    history_size = int(history_len or getattr(agent, "history_len", 24))
    if mode not in {"filters", "history", "current"}:
        raise ValueError("observation_mode must be 'filters', 'history', or 'current'")
    event_times: deque[float] = deque(maxlen=history_size)

    def policy_state(raw_state: np.ndarray, time: float) -> np.ndarray:
        if mode != "history":
            return raw_state.copy()
        recencies = np.zeros(history_size, dtype=np.float64)
        if event_times:
            ages = np.clip(time - np.asarray(event_times), 0.0, env_cfg.horizon)
            recencies[-len(event_times) :] = np.exp(-ages / env_cfg.horizon)
        return np.concatenate([raw_state, recencies])

    states = [policy_state(state, start_time)]
    times = [start_time]
    actions = []
    rewards = []
    jumps = []
    intensities = []
    discounted_return = 0.0

    while env.k < env_cfg.steps:
        t = env.t
        if constant_action is not None:
            action = constant_action
        elif agent is None:
            action = rng.uniform(0.0, 1.0)
        else:
            action = agent.act(t, policy_state(state, t))
            if exploration_std > 0.0:
                action += rng.normal(0.0, exploration_std)
        action = float(np.clip(action, 0.0, 1.0))
        next_state, reward_rate, done, info = env.step(action, rng)
        discounted_return += np.exp(-env_cfg.beta * (t - start_time)) * reward_rate * env_cfg.dt
        actions.append(action)
        rewards.append(reward_rate)
        jumps.append(info["jumps"])
        intensities.append(info["lambda"])
        event_time = max(start_time, env.t - 0.5 * env_cfg.dt)
        for _ in range(int(info["jumps"])):
            event_times.append(event_time)
        state = next_state
        states.append(policy_state(state, env.t))
        times.append(env.t)
        if done:
            break

    terminal = env.terminal_reward(np.asarray([states[-1]]))[0]
    discounted_return += np.exp(-env_cfg.beta * (env_cfg.horizon - start_time)) * terminal
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
    env_cfg: PowerLawHawkesConfig,
    agent,
    seed: int,
    episodes: int = 20,
    constant_action: float | None = None,
) -> dict:
    envs = [PowerLawHawkesEnv(env_cfg) for _ in range(episodes)]
    raw_states = np.stack([env.reset() for env in envs])
    rngs = [np.random.default_rng(seed + 10_000 * idx) for idx in range(episodes)]
    mode = getattr(agent, "observation_mode", "filters") if agent is not None else "filters"
    history_size = int(getattr(agent, "history_len", 24)) if agent is not None else 24
    event_times = [deque(maxlen=history_size) for _ in range(episodes)]
    returns = np.zeros(episodes, dtype=np.float64)
    action_sums = np.zeros(episodes, dtype=np.float64)
    jump_sums = np.zeros(episodes, dtype=np.float64)
    intensity_sums = np.zeros(episodes, dtype=np.float64)
    all_actions = []

    def policy_states(time: float) -> np.ndarray:
        if mode != "history":
            return raw_states
        rows = []
        for idx, state in enumerate(raw_states):
            recencies = np.zeros(history_size, dtype=np.float64)
            if event_times[idx]:
                ages = np.clip(time - np.asarray(event_times[idx]), 0.0, env_cfg.horizon)
                recencies[-len(event_times[idx]) :] = np.exp(-ages / env_cfg.horizon)
            rows.append(np.concatenate([state, recencies]))
        return np.stack(rows)

    for k in range(env_cfg.steps):
        time = k * env_cfg.dt
        observed_states = policy_states(time)
        if constant_action is not None:
            step_actions = np.full(episodes, float(constant_action), dtype=np.float64)
        elif agent is None:
            step_actions = np.asarray([rng.uniform(0.0, 1.0) for rng in rngs])
        elif hasattr(agent, "features") and hasattr(agent, "actor"):
            features = agent.features(np.full(episodes, time), observed_states)
            step_actions = agent.actor.forward(features).reshape(-1)
        else:
            step_actions = np.asarray([agent.act(time, state) for state in observed_states])
        step_actions = np.clip(step_actions, 0.0, 1.0)

        next_states = []
        for idx, env in enumerate(envs):
            state, reward_rate, _, info = env.step(float(step_actions[idx]), rngs[idx])
            next_states.append(state)
            returns[idx] += np.exp(-env_cfg.beta * time) * reward_rate * env_cfg.dt
            action_sums[idx] += step_actions[idx]
            jump_sums[idx] += info["jumps"]
            intensity_sums[idx] += info["lambda"]
            event_time = max(0.0, env.t - 0.5 * env_cfg.dt)
            for _ in range(int(info["jumps"])):
                event_times[idx].append(event_time)
        all_actions.extend(step_actions.tolist())
        raw_states = np.stack(next_states)

    terminal = PowerLawHawkesEnv(env_cfg).terminal_reward(raw_states)
    returns += np.exp(-env_cfg.beta * env_cfg.horizon) * terminal
    action_arr = np.asarray(all_actions, dtype=np.float64)
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_se": float(np.std(returns) / np.sqrt(max(1, episodes))),
        "return_ci90": float(1.6448536269514722 * np.std(returns) / np.sqrt(max(1, episodes))),
        "action_mean": float(np.mean(action_sums / env_cfg.steps)),
        "action_std": float(np.std(action_arr)),
        "action_min": float(np.min(action_arr)),
        "action_max": float(np.max(action_arr)),
        "jumps_mean": float(np.mean(jump_sums)),
        "intensity_mean": float(np.mean(intensity_sums / env_cfg.steps)),
        "episodes": int(episodes),
    }


def train_powerlaw_ct_ddpg(
    seed: int = 19,
    episodes: int = 120,
    warmup_episodes: int = 25,
    updates_per_episode: int = 8,
    eval_every: int = 30,
    eval_episodes: int = 60,
    eval_seed: int | None = None,
    early_stop_patience: int | None = None,
    restore_best: bool = True,
    env_cfg: PowerLawHawkesConfig | None = None,
    agent_cfg: AgentConfig | None = None,
    initial_current_agent: CTDDPGPowerLawAgent | None = None,
) -> Tuple[CTDDPGPowerLawAgent, ReplayBuffer, List[dict], dict]:
    env_cfg = env_cfg or PowerLawHawkesConfig()
    agent_cfg = agent_cfg or AgentConfig()
    rng = np.random.default_rng(seed)
    agent = CTDDPGPowerLawAgent(env_cfg, agent_cfg, rng)
    if initial_current_agent is not None:
        lift_current_agent_to_filtered(agent, initial_current_agent)
    buffer = ReplayBuffer(max_episodes=agent_cfg.replay_max_episodes)
    # Static a=0.30 is evaluated once by the experiment driver, not once per CT variant.
    baselines: dict = {}

    history: List[dict] = []
    latest_metrics: dict = {}
    best_eval_return = -np.inf
    best_state = snapshot_agent(agent)
    best_episode = 0
    evaluations_without_improvement = 0
    checkpoint_seed = seed + 1000 if eval_seed is None else int(eval_seed)
    for ep in range(1, episodes + 1):
        frac = min(1.0, ep / max(1, episodes))
        exploration = max(0.005, 0.20 * (1.0 - frac))
        learning_progress = np.clip(
            (ep - warmup_episodes) / max(1, episodes - warmup_episodes),
            0.0,
            1.0,
        )
        final_scale = float(np.clip(agent_cfg.actor_lr_final_scale, 0.0, 1.0))
        lr_scale = final_scale + (1.0 - final_scale) * 0.5 * (
            1.0 + np.cos(np.pi * learning_progress)
        )
        agent.actor_opt.lr = agent_cfg.lr_actor * lr_scale
        collector = None if ep <= warmup_episodes else agent
        episode = collect_episode(
            env_cfg,
            collector,
            rng,
            exploration_std=exploration,
            observation_mode=agent.observation_mode,
            history_len=agent.history_len,
        )
        buffer.add(episode)
        if len(buffer) >= warmup_episodes:
            latest_metrics = agent.update(buffer, rng, updates_per_episode)

        if ep == 1 or ep % eval_every == 0 or ep == episodes:
            eval_stats = evaluate_agent(env_cfg, agent, checkpoint_seed, episodes=eval_episodes)
            # Episode 1 is the untouched initialized policy. Keeping it eligible
            # prevents noisy post-warmup actor updates from forcing a worse model.
            checkpoint_eligible = ep == 1 or ep >= warmup_episodes + eval_every
            is_best = checkpoint_eligible and eval_stats["return_mean"] > best_eval_return
            if is_best:
                best_eval_return = eval_stats["return_mean"]
                best_state = snapshot_agent(agent)
                best_episode = ep
                evaluations_without_improvement = 0
            elif checkpoint_eligible:
                evaluations_without_improvement += 1
            row = {
                "episode": ep,
                "train_return": episode.discounted_return,
                "exploration_std": exploration,
                "actor_learning_rate": agent.actor_opt.lr,
                "checkpoint_eligible": float(checkpoint_eligible),
                "is_best_checkpoint": float(is_best),
                "best_episode": best_episode,
                **latest_metrics,
                **{f"eval_{k}": v for k, v in eval_stats.items()},
            }
            history.append(row)
            print(format_log_row(row))
            if (
                early_stop_patience is not None
                and early_stop_patience > 0
                and ep >= warmup_episodes
                and evaluations_without_improvement >= early_stop_patience
            ):
                print(
                    f"early stopping at episode {ep}; no validation improvement for "
                    f"{evaluations_without_improvement} evaluations (best episode {best_episode})"
                )
                break
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
