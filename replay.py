from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Episode:
    times: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    reward_rates: np.ndarray
    jumps: np.ndarray
    intensities: np.ndarray
    discounted_return: float


class ReplayBuffer:
    def __init__(self, max_episodes: int = 500):
        self.max_episodes = max_episodes
        self.episodes: List[Episode] = []

    def add(self, episode: Episode) -> None:
        self.episodes.append(episode)
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def __len__(self) -> int:
        return len(self.episodes)

    def sample_segments(self, batch_size: int, segment_len: int, rng: np.random.Generator):
        eps = rng.choice(len(self.episodes), size=batch_size, replace=True)
        times = []
        states = []
        actions = []
        rewards = []
        terminal = []
        for ep_idx in eps:
            ep = self.episodes[int(ep_idx)]
            max_start = len(ep.actions) - segment_len
            start = int(rng.integers(0, max_start + 1))
            end = start + segment_len
            times.append(ep.times[start : end + 1])
            states.append(ep.states[start : end + 1])
            actions.append(ep.actions[start:end])
            rewards.append(ep.reward_rates[start:end])
            terminal.append(end == len(ep.actions))
        return (
            np.asarray(times),
            np.asarray(states),
            np.asarray(actions)[..., None],
            np.asarray(rewards),
            np.asarray(terminal, dtype=bool),
        )

    def sample_states(self, batch_size: int, rng: np.random.Generator, terminal: bool = False):
        eps = rng.choice(len(self.episodes), size=batch_size, replace=True)
        times = []
        states = []
        for ep_idx in eps:
            ep = self.episodes[int(ep_idx)]
            if terminal:
                idx = len(ep.states) - 1
            else:
                idx = int(rng.integers(0, len(ep.states) - 1))
            times.append(ep.times[idx])
            states.append(ep.states[idx])
        return np.asarray(times), np.asarray(states)
