import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FullBandit(gym.Env):
    metadata = {
        'render_modes': ['human']
    }

    def __init__(self,
                 n_arms: int,
                 action_size: int,
                 rewards: np.ndarray,
                 horizon: int | None = None,
                 winner_feedback: bool = False):
        if n_arms < 2:
            raise ValueError("Invalid number of arms")
        self.n_arms = n_arms

        if not 1 <= action_size < n_arms:
            raise ValueError("Invalid action size")
        self.action_size = action_size

        self.rewards = np.asarray(rewards)
        if (self.rewards.ndim != 2
                or self.rewards.shape[1] != self.n_arms
                or self.rewards.shape[0] == 0):
            raise ValueError("Invalid reward matrix")
        if not np.all(np.isfinite(self.rewards)) or np.any(
                (self.rewards < 0) | (self.rewards > 1)):
            raise ValueError("Invalid rewards")

        self.horizon = self.rewards.shape[0] if horizon is None else horizon
        if not 1 <= self.horizon <= self.rewards.shape[0]:
            raise ValueError("Invalid horizon")

        self.winner_feedback = winner_feedback

        self.observation_space = spaces.Box(
            low=0., high=float(self.action_size), shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.MultiBinary(self.n_arms)

        self.t = 1
        self._episode_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 1
        self._episode_over = False
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: np.ndarray, render=False):
        if self._episode_over:
            raise RuntimeError("Episode is over; call reset before stepping again")
        if (not self.action_space.contains(action)
                or np.sum(action) != self.action_size):
            raise ValueError("Invalid action")

        reward = float(np.dot(action, self.rewards[self.t - 1]))
        observation = np.array([reward], dtype=np.float32)

        info = {}
        if self.winner_feedback:
            candidates = np.flatnonzero(action)
            attractions = self.rewards[self.t - 1][candidates]
            normalization = attractions.sum()
            if np.isclose(normalization, 0):
                winner = self.np_random.choice(candidates)
            else:
                winner = self.np_random.choice(
                    candidates, p=attractions / normalization
                )
            info["winner"] = int(winner)

        terminated = False
        truncated = self.t >= self.horizon
        if truncated:
            self._episode_over = True
        else:
            self.t += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human', close=False):
        print(np.array2string(self.rewards[self.t - 1]))
