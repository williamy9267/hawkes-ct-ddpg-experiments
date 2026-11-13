from __future__ import annotations

import sys
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

HERE = Path(__file__).resolve().parent
PROJECT_HELPER_ROOT = next(
    (
        parent
        for parent in (HERE, *HERE.parents)
        if (parent / "replay.py").exists()
    ),
    HERE,
)
for path in (PROJECT_HELPER_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from powerlaw_env import PowerLawHawkesConfig, PowerLawHawkesEnv
from ct_ddpg_powerlaw import evaluate_agent
from replay import Episode


_FILTER_SUMMARY_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _filter_summary_cache_key(cfg: PowerLawHawkesConfig) -> tuple:
    return tuple(sorted((name, repr(value)) for name, value in vars(cfg).items()))


def _filter_summary(filters: np.ndarray, cfg: PowerLawHawkesConfig) -> np.ndarray:
    """Generic level and age-shape summaries of the observed filter curve."""

    betas = np.asarray(cfg.filter_betas, dtype=np.float64)
    levels = np.log1p(betas * np.maximum(filters, 0.0))
    gaps = np.diff(betas)
    midpoints = 0.5 * (betas[:-1] + betas[1:])
    age_levels = np.log1p(
        midpoints * np.maximum(filters[:-1] - filters[1:], 0.0) / gaps
    )
    return np.asarray(
        [levels.mean(), levels.std(), age_levels.mean(), age_levels.max()],
        dtype=np.float64,
    )


def fit_observed_filter_summary_stats(
    cfg: PowerLawHawkesConfig,
    episodes: int = 40,
    seed: int = 31_415,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize summaries using deterministic random-policy rollouts."""

    key = _filter_summary_cache_key(cfg)
    if key in _FILTER_SUMMARY_CACHE:
        return _FILTER_SUMMARY_CACHE[key]

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(episodes):
        env = PowerLawHawkesEnv(cfg)
        state = env.reset()
        for _ in range(cfg.steps):
            samples.append(_filter_summary(state[1:], cfg))
            state, _, done, _ = env.step(float(rng.uniform(0.0, 1.0)), rng)
            if done:
                break
    values = np.asarray(samples, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1e-6)
    _FILTER_SUMMARY_CACHE[key] = (mean, scale)
    return mean, scale


def sb3_available() -> bool:
    try:
        import stable_baselines3  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class SB3Config:
    total_timesteps: int = 60_000
    history_len: int = 24
    feature_mode: str = "matched"
    n_envs: int = 4
    ddpg_n_envs: int = 4
    sac_learning_rate: float = 3e-4
    ddpg_learning_rate: float = 2e-4
    ppo_learning_rate: float = 3e-4
    sac_batch_size: int = 256
    ddpg_batch_size: int = 256
    sac_gradient_steps: int = 2
    sac_n_steps: int = 1
    ddpg_gradient_steps: int = 2
    sac_tau: float = 0.01
    ddpg_tau: float = 0.01
    sac_ent_coef: str | float = "auto_0.05"
    gamma: float | None = None
    reward_scale: float = 25.0
    learning_starts: int = 2_500
    ddpg_action_noise_sigma: float = 0.15
    ddpg_initial_action: float | None = None
    policy_hidden: Tuple[int, int] = (128, 128)
    optimizer_weight_decay: float = 0.0
    warm_start_critic: bool = True
    warm_start_memory_noise: float = 0.0
    ppo_n_steps: int = 256
    ppo_batch_size: int = 256
    ppo_n_epochs: int = 10
    ppo_ent_coef: float = 0.0
    ppo_log_std_init: float = -1.2
    ppo_gae_lambda: float = 0.97
    ppo_clip_range: float = 0.2
    ppo_target_kl: float | None = 0.05
    ppo_vf_coef: float = 0.7
    ppo_normalize_reward: bool = False
    ppo_clip_reward: float = 10.0
    eval_freq: int = 10_000
    eval_episodes: int = 24
    eval_n_envs: int = 8
    early_stop_patience: int = 3
    early_stop_min_evals: int = 1
    restore_best: bool = True
    seed: int = 77
    device: str = "cpu"
    ppo_device: str = "cpu"
    torch_num_threads: int = 2


class PowerLawHawkesGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env_cfg: PowerLawHawkesConfig | None = None,
        observation_mode: str = "history",
        history_len: int = 24,
        seed: int | None = None,
        reward_scale: float = 1.0,
        feature_mode: str = "matched",
    ):
        super().__init__()
        self.env_cfg = env_cfg or PowerLawHawkesConfig()
        self.observation_mode = observation_mode
        self.history_len = int(history_len)
        self.reward_scale = float(reward_scale)
        if feature_mode not in {"summary", "matched", "conditioned", "legacy"}:
            raise ValueError("feature_mode must be 'summary', 'matched', 'conditioned', or 'legacy'")
        self.feature_mode = feature_mode
        self.rng = np.random.default_rng(seed)
        self.env = PowerLawHawkesEnv(self.env_cfg)
        if self.observation_mode == "filters" and self.feature_mode == "summary":
            self.filter_summary_mean, self.filter_summary_scale = fit_observed_filter_summary_stats(
                self.env_cfg
            )
        else:
            self.filter_summary_mean = None
            self.filter_summary_scale = None
        # Keep the algorithmic action symmetric, then map it to the physical
        # Hawkes control interval [0, 1] inside step().
        self.action_space = spaces.Box(low=-np.ones(1, dtype=np.float32), high=np.ones(1, dtype=np.float32))
        obs_dim = self._obs_dim()
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        self.jump_hist = deque(maxlen=self.history_len)
        self.action_hist = deque(maxlen=self.history_len)
        self.x_hist = deque(maxlen=self.history_len)

    def _obs_dim(self) -> int:
        if self.observation_mode == "filters":
            if self.feature_mode == "summary":
                return 6
            return 2 + len(self.env_cfg.filter_betas)
        if self.observation_mode == "current":
            return 2
        if self.observation_mode == "history":
            if self.feature_mode != "legacy":
                return 2 + self.history_len
            return 2 + 3 * self.history_len
        raise ValueError(f"unknown observation_mode {self.observation_mode!r}")

    def _reset_history(self) -> None:
        self.jump_hist.clear()
        self.action_hist.clear()
        self.x_hist.clear()
        if self.feature_mode == "legacy":
            for _ in range(self.history_len):
                self.jump_hist.append(0.0)
                self.action_hist.append(0.0)
                self.x_hist.append(self.env_cfg.x0)

    def _event_history(self) -> np.ndarray:
        recency = np.zeros(self.history_len, dtype=np.float64)
        n_events = len(self.jump_hist)
        if n_events:
            event_times = np.asarray(self.jump_hist, dtype=np.float64)
            ages = np.clip(self.env.t - event_times, 0.0, self.env_cfg.horizon)
            recency[-n_events:] = np.exp(-ages / self.env_cfg.horizon)
        return recency

    def _obs(self, state: np.ndarray) -> np.ndarray:
        cfg = self.env_cfg
        t_feat = 2.0 * self.env.t / cfg.horizon - 1.0
        x_feat = np.clip(state[0] / cfg.x_scale, -5.0, 5.0)
        if self.observation_mode == "filters":
            betas = np.asarray(cfg.filter_betas, dtype=np.float64)
            filters = np.maximum(state[1:], 0.0)
            levels = np.log1p(betas * filters)
            if self.feature_mode == "summary":
                summary = _filter_summary(filters, cfg)
                z_feat = (summary - self.filter_summary_mean) / self.filter_summary_scale
                z_feat = np.clip(z_feat, -5.0, 5.0)
            elif self.feature_mode == "matched":
                z_feat = np.clip(filters / cfg.z_scale, 0.0, 5.0)
            elif self.feature_mode == "conditioned" and levels.size > 1:
                slopes = (levels[:-1] - levels[1:]) / np.diff(betas)
                z_feat = np.concatenate([levels[:1], slopes])
                z_feat = np.clip(z_feat, -5.0, 5.0)
            else:
                z_feat = np.clip(levels, 0.0, 5.0)
            obs = np.concatenate([[t_feat, x_feat], z_feat])
        elif self.observation_mode == "current":
            obs = np.asarray([t_feat, x_feat], dtype=np.float64)
        elif self.observation_mode == "history":
            if self.feature_mode != "legacy":
                jump_hist = self._event_history()
                obs = np.concatenate([[t_feat, x_feat], jump_hist])
            else:
                jump_hist = np.asarray(self.jump_hist, dtype=np.float64) / max(1.0, cfg.baseline_max * cfg.dt)
                action_hist = np.asarray(self.action_hist, dtype=np.float64)
                x_hist = np.asarray(self.x_hist, dtype=np.float64)
                x_hist = x_hist / cfg.x_scale
                obs = np.concatenate([[t_feat, x_feat], jump_hist, action_hist, x_hist])
        else:
            raise ValueError(f"unknown observation_mode {self.observation_mode!r}")
        return np.asarray(obs, dtype=np.float32)

    @staticmethod
    def physical_action(action: np.ndarray | float) -> float:
        internal = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        return 0.5 * (internal + 1.0)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        state = self.env.reset()
        self._reset_history()
        return self._obs(state), {}

    def step(self, action):
        action_scalar = self.physical_action(action)
        state, reward_rate, done, info = self.env.step(action_scalar, self.rng)
        if self.observation_mode == "history":
            if self.feature_mode != "legacy":
                event_time = max(0.0, self.env.t - 0.5 * self.env_cfg.dt)
                for _ in range(int(info["jumps"])):
                    self.jump_hist.append(event_time)
            else:
                self.jump_hist.append(float(info["jumps"]))
                self.action_hist.append(action_scalar)
                self.x_hist.append(float(state[0]))
        reward = reward_rate * self.env_cfg.dt
        if done:
            reward += self.env.terminal_reward(np.asarray([state]))[0]
        info = dict(info)
        info["physical_action"] = action_scalar
        return self._obs(state), float(self.reward_scale * reward), bool(done), False, info


class SB3PolicyAdapter:
    def __init__(
        self,
        model,
        observation_mode: str,
        history_len: int,
        env_cfg: PowerLawHawkesConfig,
        feature_mode: str = "matched",
    ):
        self.model = model
        self.obs_env = PowerLawHawkesGymEnv(
            env_cfg,
            observation_mode=observation_mode,
            history_len=history_len,
            feature_mode=feature_mode,
        )
        self.obs_env.reset()

    def act(self, time: float, state: np.ndarray) -> float:
        self.obs_env.env.set_state(time, state)
        obs = self.obs_env._obs(state)
        action, _ = self.model.predict(obs, deterministic=True)
        return self.obs_env.physical_action(action)


def make_env(
    env_cfg: PowerLawHawkesConfig,
    observation_mode: str,
    history_len: int,
    seed: int,
    reward_scale: float,
    feature_mode: str = "matched",
) -> Callable[[], gym.Env]:
    def _factory():
        return PowerLawHawkesGymEnv(
            env_cfg,
            observation_mode=observation_mode,
            history_len=history_len,
            seed=seed,
            reward_scale=reward_scale,
            feature_mode=feature_mode,
        )

    return _factory


def transfer_ddpg_current_state_weights(source_model, target_model, current_obs_dim: int = 2) -> None:
    """Embed a current-state DDPG policy in a larger observation space."""

    import torch

    def copy_actor(source_actor, target_actor) -> None:
        source_layers = [module for module in source_actor.mu if isinstance(module, torch.nn.Linear)]
        target_layers = [module for module in target_actor.mu if isinstance(module, torch.nn.Linear)]
        if len(source_layers) != len(target_layers):
            raise ValueError("source and target DDPG actors must use the same MLP architecture")
        with torch.no_grad():
            target_layers[0].weight.zero_()
            target_layers[0].weight[:, :current_obs_dim].copy_(source_layers[0].weight[:, :current_obs_dim])
            target_layers[0].bias.copy_(source_layers[0].bias)
            for source_layer, target_layer in zip(source_layers[1:], target_layers[1:]):
                target_layer.weight.copy_(source_layer.weight)
                target_layer.bias.copy_(source_layer.bias)

    def copy_critic(source_critic, target_critic) -> None:
        source_layers = [module for module in source_critic.qf0 if isinstance(module, torch.nn.Linear)]
        target_layers = [module for module in target_critic.qf0 if isinstance(module, torch.nn.Linear)]
        if len(source_layers) != len(target_layers):
            raise ValueError("source and target DDPG critics must use the same MLP architecture")
        with torch.no_grad():
            target_layers[0].weight.zero_()
            target_layers[0].weight[:, :current_obs_dim].copy_(source_layers[0].weight[:, :current_obs_dim])
            target_layers[0].weight[:, -1].copy_(source_layers[0].weight[:, -1])
            target_layers[0].bias.copy_(source_layers[0].bias)
            for source_layer, target_layer in zip(source_layers[1:], target_layers[1:]):
                target_layer.weight.copy_(source_layer.weight)
                target_layer.bias.copy_(source_layer.bias)

    copy_actor(source_model.actor, target_model.actor)
    copy_actor(source_model.actor_target, target_model.actor_target)
    copy_critic(source_model.critic, target_model.critic)
    copy_critic(source_model.critic_target, target_model.critic_target)


def transfer_sac_current_state_weights(
    source_model,
    target_model,
    current_obs_dim: int = 2,
    copy_critics: bool = True,
    memory_weight_scale: float = 0.0,
) -> None:
    """Embed a current-state SAC policy and critics in a larger observation space."""

    import torch

    def linear_layers(sequence) -> list[torch.nn.Linear]:
        return [module for module in sequence if isinstance(module, torch.nn.Linear)]

    def copy_actor(source_actor, target_actor) -> None:
        source_layers = linear_layers(source_actor.latent_pi)
        target_layers = linear_layers(target_actor.latent_pi)
        if len(source_layers) != len(target_layers):
            raise ValueError("source and target SAC actors must use the same MLP architecture")
        with torch.no_grad():
            target_layers[0].weight.zero_()
            target_layers[0].weight[:, :current_obs_dim].copy_(source_layers[0].weight[:, :current_obs_dim])
            if memory_weight_scale > 0.0:
                target_layers[0].weight[:, current_obs_dim:].normal_(
                    mean=0.0,
                    std=float(memory_weight_scale),
                )
            target_layers[0].bias.copy_(source_layers[0].bias)
            for source_layer, target_layer in zip(source_layers[1:], target_layers[1:]):
                target_layer.weight.copy_(source_layer.weight)
                target_layer.bias.copy_(source_layer.bias)
            target_actor.mu.load_state_dict(source_actor.mu.state_dict())
            target_actor.log_std.load_state_dict(source_actor.log_std.state_dict())

    def copy_q_network(source_q, target_q) -> None:
        source_layers = linear_layers(source_q)
        target_layers = linear_layers(target_q)
        if len(source_layers) != len(target_layers):
            raise ValueError("source and target SAC critics must use the same MLP architecture")
        with torch.no_grad():
            target_layers[0].weight.zero_()
            target_layers[0].weight[:, :current_obs_dim].copy_(source_layers[0].weight[:, :current_obs_dim])
            target_layers[0].weight[:, -1].copy_(source_layers[0].weight[:, -1])
            target_layers[0].bias.copy_(source_layers[0].bias)
            for source_layer, target_layer in zip(source_layers[1:], target_layers[1:]):
                target_layer.weight.copy_(source_layer.weight)
                target_layer.bias.copy_(source_layer.bias)

    copy_actor(source_model.actor, target_model.actor)
    if copy_critics:
        for source_q, target_q in zip(source_model.critic.q_networks, target_model.critic.q_networks):
            copy_q_network(source_q, target_q)
        for source_q, target_q in zip(source_model.critic_target.q_networks, target_model.critic_target.q_networks):
            copy_q_network(source_q, target_q)


def train_sb3_baselines(
    env_cfg: PowerLawHawkesConfig,
    observation_mode: str = "history",
    cfg: SB3Config | None = None,
    algorithms: Tuple[str, ...] = ("sac", "ddpg"),
    warm_start_models: dict[str, object] | None = None,
) -> dict[str, object]:
    if not sb3_available():
        raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")

    import torch
    from stable_baselines3 import DDPG, PPO, SAC
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cfg = cfg or SB3Config()
    torch.set_num_threads(max(1, int(cfg.torch_num_threads)))
    device = cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    ppo_device = cfg.ppo_device if cfg.ppo_device != "cuda" or torch.cuda.is_available() else "cpu"
    gamma = float(cfg.gamma if cfg.gamma is not None else np.exp(-env_cfg.beta * env_cfg.dt))
    learning_starts = min(cfg.learning_starts, max(1, cfg.total_timesteps // 4))

    class CommonRandomEvalCallback(EvalCallback):
        """Re-use validation random-number streams at every checkpoint."""

        def __init__(self, *args, validation_seed: int, **kwargs):
            super().__init__(*args, **kwargs)
            self.validation_seed = int(validation_seed)

        def _on_step(self) -> bool:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                self.eval_env.seed(self.validation_seed)
            return super()._on_step()

    def build_vec_env(n_envs: int, seed_offset: int = 0):
        return DummyVecEnv(
            [
                lambda i=i: Monitor(
                    PowerLawHawkesGymEnv(
                        env_cfg,
                        observation_mode=observation_mode,
                        history_len=cfg.history_len,
                        seed=cfg.seed + seed_offset + i,
                        reward_scale=cfg.reward_scale,
                        feature_mode=cfg.feature_mode,
                    )
                )
                for i in range(n_envs)
            ]
        )

    def build_ppo_env(n_envs: int, seed_offset: int, training: bool):
        env = build_vec_env(n_envs, seed_offset=seed_offset)
        if not cfg.ppo_normalize_reward:
            return env
        return VecNormalize(
            env,
            training=training,
            norm_obs=False,
            norm_reward=training,
            gamma=gamma,
            clip_reward=cfg.ppo_clip_reward,
        )

    common = dict(
        policy="MlpPolicy",
        verbose=0,
        gamma=gamma,
        seed=cfg.seed,
    )
    optimizer_kwargs = {"weight_decay": max(0.0, float(cfg.optimizer_weight_decay))}
    builders = {
        "sac": lambda: SAC(
            **common,
            env=build_vec_env(cfg.n_envs, seed_offset=0),
            device=device,
            learning_rate=cfg.sac_learning_rate,
            batch_size=cfg.sac_batch_size,
            buffer_size=max(100_000, cfg.total_timesteps),
            policy_kwargs={"net_arch": list(cfg.policy_hidden), "optimizer_kwargs": optimizer_kwargs},
            train_freq=1,
            gradient_steps=cfg.sac_gradient_steps,
            n_steps=max(1, int(cfg.sac_n_steps)),
            learning_starts=learning_starts,
            tau=cfg.sac_tau,
            ent_coef=cfg.sac_ent_coef,
        ),
        "ddpg": lambda: DDPG(
            **common,
            env=build_vec_env(cfg.ddpg_n_envs, seed_offset=10_000),
            device=device,
            learning_rate=cfg.ddpg_learning_rate,
            batch_size=cfg.ddpg_batch_size,
            buffer_size=max(100_000, cfg.total_timesteps),
            policy_kwargs={"net_arch": list(cfg.policy_hidden), "optimizer_kwargs": optimizer_kwargs},
            train_freq=1,
            gradient_steps=cfg.ddpg_gradient_steps,
            learning_starts=learning_starts,
            tau=cfg.ddpg_tau,
            action_noise=NormalActionNoise(mean=np.zeros(1), sigma=cfg.ddpg_action_noise_sigma * np.ones(1)),
        ),
        "ppo": lambda: PPO(
            **common,
            env=build_ppo_env(cfg.n_envs, seed_offset=20_000, training=True),
            device=ppo_device,
            learning_rate=cfg.ppo_learning_rate,
            policy_kwargs={
                "net_arch": list(cfg.policy_hidden),
                "log_std_init": cfg.ppo_log_std_init,
            },
            n_steps=cfg.ppo_n_steps,
            batch_size=cfg.ppo_batch_size,
            n_epochs=cfg.ppo_n_epochs,
            ent_coef=cfg.ppo_ent_coef,
            gae_lambda=cfg.ppo_gae_lambda,
            clip_range=cfg.ppo_clip_range,
            target_kl=cfg.ppo_target_kl,
            vf_coef=cfg.ppo_vf_coef,
        ),
    }
    unknown = sorted(set(algorithms) - set(builders))
    if unknown:
        raise ValueError(f"unknown SB3 algorithms: {', '.join(unknown)}")
    requested = tuple(dict.fromkeys(algorithms))
    warm_start_models = warm_start_models or {}
    trained = {}
    n_envs_by_name = {"sac": cfg.n_envs, "ddpg": cfg.ddpg_n_envs, "ppo": cfg.n_envs}
    seed_offsets = {"sac": 0, "ddpg": 10_000, "ppo": 20_000}
    for name in requested:
        model = builders[name]()
        if name == "ddpg" and name not in warm_start_models and cfg.ddpg_initial_action is not None:
            physical_action = float(np.clip(cfg.ddpg_initial_action, 1e-4, 1.0 - 1e-4))
            internal_action = 2.0 * physical_action - 1.0
            final_linear = [layer for layer in model.actor.mu if isinstance(layer, torch.nn.Linear)][-1]
            with torch.no_grad():
                final_linear.weight.zero_()
                final_linear.bias.fill_(float(np.arctanh(np.clip(internal_action, -0.999, 0.999))))
                model.actor_target.load_state_dict(model.actor.state_dict())
            print(f"initialized DDPG actor at constant physical action {physical_action:.3f}")
        if name in warm_start_models:
            if name == "ddpg":
                transfer_ddpg_current_state_weights(warm_start_models[name], model)
            elif name == "sac":
                transfer_sac_current_state_weights(
                    warm_start_models[name],
                    model,
                    copy_critics=cfg.warm_start_critic,
                    memory_weight_scale=cfg.warm_start_memory_noise,
                )
            else:
                raise ValueError("nested-state warm starts are implemented for SAC and DDPG only")
            print(f"initialized {name.upper()} from a current-state policy with zero memory weights")
        n_envs = cfg.ddpg_n_envs if name == "ddpg" else cfg.n_envs
        algorithm_device = ppo_device if name == "ppo" else device
        print(
            f"training {name.upper()} with observation_mode={observation_mode}, "
            f"gamma={gamma:.6f}, reward_scale={cfg.reward_scale:.1f}, n_envs={n_envs}, device={algorithm_device}"
        )
        if name == "ppo":
            eval_env = build_ppo_env(
                cfg.eval_n_envs,
                seed_offset=500_000 + seed_offsets[name],
                training=False,
            )
        else:
            eval_env = build_vec_env(cfg.eval_n_envs, seed_offset=500_000 + seed_offsets[name])
        eval_freq_calls = max(1, int(cfg.eval_freq) // max(1, n_envs_by_name[name]))
        validation_seed = cfg.seed + 500_000 + seed_offsets[name]
        with tempfile.TemporaryDirectory(prefix=f"powerlaw_{name}_{observation_mode}_") as checkpoint_dir:
            stop_callback = None
            if cfg.early_stop_patience > 0:
                stop_callback = StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=int(cfg.early_stop_patience),
                    min_evals=max(0, int(cfg.early_stop_min_evals)),
                    verbose=1,
                )
            callback = CommonRandomEvalCallback(
                eval_env,
                validation_seed=validation_seed,
                callback_after_eval=stop_callback,
                best_model_save_path=checkpoint_dir if cfg.restore_best else None,
                log_path=checkpoint_dir,
                n_eval_episodes=cfg.eval_episodes,
                eval_freq=eval_freq_calls,
                deterministic=True,
                verbose=1,
            )
            evaluation_history = []
            if name in warm_start_models:
                eval_env.seed(validation_seed)
                initial_rewards, _ = evaluate_policy(
                    model,
                    eval_env,
                    n_eval_episodes=cfg.eval_episodes,
                    deterministic=True,
                    return_episode_rewards=True,
                    warn=False,
                )
                initial_mean = float(np.mean(initial_rewards))
                initial_std = float(np.std(initial_rewards))
                callback.best_mean_reward = initial_mean
                model.save(Path(checkpoint_dir) / "best_model")
                evaluation_history.append(
                    {
                        "timesteps": 0,
                        "scaled_return_mean": initial_mean,
                        "scaled_return_std": initial_std,
                        "cost_mean": -initial_mean / cfg.reward_scale,
                    }
                )
                print(f"warm-start scaled eval reward={initial_mean:.4f}")
            model.learn(total_timesteps=cfg.total_timesteps, callback=callback, progress_bar=False)
            evaluation_history.extend(
                {
                    "timesteps": int(timesteps),
                    "scaled_return_mean": float(np.mean(episode_rewards)),
                    "scaled_return_std": float(np.std(episode_rewards)),
                    "cost_mean": float(-np.mean(episode_rewards) / cfg.reward_scale),
                }
                for timesteps, episode_rewards in zip(
                    callback.evaluations_timesteps,
                    callback.evaluations_results,
                )
            )
            best_path = Path(checkpoint_dir) / "best_model.zip"
            if cfg.restore_best and best_path.exists():
                model = type(model).load(best_path, env=model.get_env(), device=algorithm_device)
                print(f"restored best {name.upper()} checkpoint; scaled eval reward={callback.best_mean_reward:.4f}")
            model.powerlaw_evaluation_history = evaluation_history
        eval_env.close()
        trained[name] = model
    return trained


def evaluate_sb3_models(
    env_cfg: PowerLawHawkesConfig,
    models: dict[str, object],
    observation_mode: str,
    history_len: int,
    seed: int,
    episodes: int,
    feature_mode: str = "matched",
) -> dict[str, dict]:
    rows = {}
    for name, model in models.items():
        envs = [
            PowerLawHawkesGymEnv(
                env_cfg,
                observation_mode=observation_mode,
                history_len=history_len,
                seed=seed + 10_000 * ep_idx,
                feature_mode=feature_mode,
            )
            for ep_idx in range(episodes)
        ]
        observations = np.stack([env.reset()[0] for env in envs])
        returns = np.zeros(episodes, dtype=np.float64)
        actions_by_episode = [[] for _ in envs]
        jumps_by_episode = [[] for _ in envs]
        intensities_by_episode = [[] for _ in envs]
        for k in range(env_cfg.steps):
            actions, _ = model.predict(observations, deterministic=True)
            next_observations = []
            for ep_idx, env in enumerate(envs):
                obs, reward, done, _, info = env.step(actions[ep_idx])
                next_observations.append(obs)
                action_scalar = float(info["physical_action"])
                t = k * env_cfg.dt
                if done:
                    terminal = env.env.terminal_reward(np.asarray([env.env.observation()]))[0]
                    step_reward_rate_part = reward - terminal
                    returns[ep_idx] += np.exp(-env_cfg.beta * t) * step_reward_rate_part
                    returns[ep_idx] += np.exp(-env_cfg.beta * env_cfg.horizon) * terminal
                else:
                    returns[ep_idx] += np.exp(-env_cfg.beta * t) * reward
                actions_by_episode[ep_idx].append(action_scalar)
                jumps_by_episode[ep_idx].append(float(info["jumps"]))
                intensities_by_episode[ep_idx].append(float(info["lambda"]))
            observations = np.stack(next_observations)

        episode_action_means = [float(np.mean(values)) for values in actions_by_episode]
        all_actions = [value for values in actions_by_episode for value in values]
        jumps = [float(np.sum(values)) for values in jumps_by_episode]
        intensities = [float(np.mean(values)) for values in intensities_by_episode]
        action_arr = np.asarray(all_actions, dtype=np.float64)
        rows[name] = {
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "return_se": float(np.std(returns) / np.sqrt(max(1, episodes))),
            "return_ci90": float(1.6448536269514722 * np.std(returns) / np.sqrt(max(1, episodes))),
            "action_mean": float(np.mean(episode_action_means)),
            "action_std": float(np.std(action_arr)) if action_arr.size else float("nan"),
            "action_min": float(np.min(action_arr)) if action_arr.size else float("nan"),
            "action_max": float(np.max(action_arr)) if action_arr.size else float("nan"),
            "jumps_mean": float(np.mean(jumps)),
            "intensity_mean": float(np.mean(intensities)),
            "episodes": int(episodes),
        }
    return rows


def collect_sb3_episode(
    env_cfg: PowerLawHawkesConfig,
    model,
    observation_mode: str,
    history_len: int,
    seed: int,
    feature_mode: str = "matched",
) -> Episode:
    """Roll out one SB3 policy while preserving the history observation wrapper."""

    gym_env = PowerLawHawkesGymEnv(
        env_cfg,
        observation_mode=observation_mode,
        history_len=history_len,
        seed=seed,
        feature_mode=feature_mode,
    )
    obs, _ = gym_env.reset()
    state = gym_env.env.observation()
    states = [state.copy()]
    times = [0.0]
    actions = []
    rewards = []
    jumps = []
    intensities = []
    discounted_return = 0.0

    for k in range(env_cfg.steps):
        t = k * env_cfg.dt
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = gym_env.step(action)
        action_scalar = float(info["physical_action"])
        next_state = gym_env.env.observation()
        reward_rate = float(info["cost_rate"] * -1.0)
        discounted_return += np.exp(-env_cfg.beta * t) * reward_rate * env_cfg.dt
        actions.append(action_scalar)
        rewards.append(reward_rate)
        jumps.append(int(info["jumps"]))
        intensities.append(float(info["lambda"]))
        states.append(next_state.copy())
        times.append((k + 1) * env_cfg.dt)
        if done:
            break

    terminal = gym_env.env.terminal_reward(np.asarray([states[-1]]))[0]
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
