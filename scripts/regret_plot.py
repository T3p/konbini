"""Utilities for plotting regret in the example scripts."""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_reward_matrix(
    n_arms: int,
    action_size: int,
    horizon: int,
    rng: np.random.Generator,
    advantage: float = 0.2,
) -> np.ndarray:
    """Generate rewards with a fixed advantage for m favored arms."""
    if n_arms < 2 or not 1 <= action_size < n_arms:
        raise ValueError("action_size must be between 1 and n_arms - 1")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 0.0 < advantage < 1.0:
        raise ValueError("advantage must lie in (0, 1)")

    rewards = rng.uniform(0.0, 1.0 - advantage, (horizon, n_arms))
    favored_arms = rng.choice(n_arms, size=action_size, replace=False)
    rewards[:, favored_arms] += advantage
    return rewards


def save_regret_plot(
    rewards: np.ndarray,
    actions_by_label: dict[str, np.ndarray],
    action_size: int,
    output_path: Path,
    title: str,
) -> tuple[int, ...]:
    """Plot cumulative regret curves against the best fixed action."""
    subsets = combinations(range(rewards.shape[1]), action_size)
    best_subset = max(
        subsets,
        key=lambda subset: rewards[:, subset].sum(),
    )
    benchmark_rewards = rewards[:, best_subset].sum(axis=1)

    figure, axes = plt.subplots()
    timesteps = np.arange(1, rewards.shape[0] + 1)
    for label, actions in actions_by_label.items():
        selected_rewards = np.sum(rewards * actions, axis=1)
        cumulative_regret = np.cumsum(
            benchmark_rewards - selected_rewards
        )
        axes.plot(timesteps, cumulative_regret, label=label)
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.set_xlabel("Timestep")
    axes.set_ylabel("Cumulative regret")
    axes.set_title(title)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return best_subset
