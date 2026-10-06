"""Reward-matrix factories for reusable bandit instances."""

from __future__ import annotations

from math import comb

import numpy as np

DEFAULT_N_BLOCKS = 4
DEFAULT_CORRELATED_N_GOOD_ARMS = 8
DEFAULT_BERNOULLI_N_GOOD_ARMS = 8


def _validate_parameters(
    n_arms: int,
    action_size: int,
    horizon: int,
    advantage: float,
) -> None:
    if n_arms < 2 or not 1 <= action_size < n_arms:
        raise ValueError("action_size must be between 1 and n_arms - 1")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 0.0 < advantage < 1.0:
        raise ValueError("advantage must lie in (0, 1)")


def stationary_good_arms(
    n_arms: int,
    action_size: int,
    horizon: int,
    seed: int | None = None,
    advantage: float = 0.1,
) -> np.ndarray:
    """Return an instance with a fixed advantage for ``action_size`` arms."""
    _validate_parameters(n_arms, action_size, horizon, advantage)

    rng = np.random.default_rng(seed)
    rewards = rng.uniform(0.0, 1.0 - advantage, (horizon, n_arms))
    good_arms = rng.choice(n_arms, size=action_size, replace=False)
    rewards[:, good_arms] = np.clip(
        rewards[:, good_arms] + advantage, 0.0, 1.0
    )
    return rewards


def correlated(
    n_arms: int,
    action_size: int,
    horizon: int,
    seed: int | None = None,
    advantage: float = 0.1,
    n_good_arms: int = DEFAULT_CORRELATED_N_GOOD_ARMS,
) -> np.ndarray:
    """Return Bernoulli arm rewards with shared uniform round noise.

    Each reward is ``0.25 * B[t, i] + U[t]``, where the Bernoulli variables
    are independent across rounds and arms, while ``U[t]`` is shared by all
    arms in round ``t``. Exactly ``n_good_arms`` arms have an expected-reward
    advantage equal to ``advantage``.
    """
    _validate_parameters(n_arms, action_size, horizon, advantage)
    if not isinstance(n_good_arms, (int, np.integer)):
        raise ValueError("n_good_arms must be an integer")
    if not 1 <= n_good_arms < n_arms:
        raise ValueError("n_good_arms must be between 1 and n_arms - 1")
    if advantage > 3.0 / 16.0:
        raise ValueError("advantage must be at most 3/16")

    rng = np.random.default_rng(seed)
    good_arms = rng.choice(n_arms, size=n_good_arms, replace=False)
    probabilities = np.full(n_arms, 0.25)
    probabilities[good_arms] += 4.0 * advantage
    bernoulli_rewards = rng.random((horizon, n_arms)) < probabilities
    shared_rewards = rng.uniform(0.25, 0.5, (horizon, 1))
    return 0.25 * bernoulli_rewards + shared_rewards


def bernoulli(
    n_arms: int,
    action_size: int,
    horizon: int,
    seed: int | None = None,
    advantage: float = 0.1,
    n_good_arms: int = DEFAULT_BERNOULLI_N_GOOD_ARMS,
) -> np.ndarray:
    """Return stationary Bernoulli rewards with random arm parameters.

    Each arm starts with an independent parameter sampled uniformly from
    ``[0, 1 - advantage]``. A random set of exactly ``n_good_arms`` arms then
    receives ``advantage`` in addition to its sampled parameter.
    """
    _validate_parameters(n_arms, action_size, horizon, advantage)
    if not isinstance(n_good_arms, (int, np.integer)):
        raise ValueError("n_good_arms must be an integer")
    if not 1 <= n_good_arms < n_arms:
        raise ValueError("n_good_arms must be between 1 and n_arms - 1")

    rng = np.random.default_rng(seed)
    probabilities = rng.uniform(0.0, 1.0 - advantage, n_arms)
    good_arms = rng.choice(n_arms, size=n_good_arms, replace=False)
    probabilities[good_arms] += advantage
    return (rng.random((horizon, n_arms)) < probabilities).astype(float)


def corrupted_stationary_good_arms(
    n_arms: int,
    action_size: int,
    horizon: int,
    seed: int | None = None,
    advantage: float = 0.2,
    corruption_probability: float = 0.1,
) -> np.ndarray:
    """Return a stationary instance with randomly corrupted decoy rounds.

    A fixed decoy subset, distinct from the good subset, receives reward one
    on each independently corrupted round.
    """
    _validate_parameters(n_arms, action_size, horizon, advantage)
    if (not np.isfinite(corruption_probability)
            or not 0.0 <= corruption_probability <= 1.0):
        raise ValueError("corruption_probability must lie in [0, 1]")

    rng = np.random.default_rng(seed)
    good_arms = tuple(
        sorted(rng.choice(n_arms, size=action_size, replace=False))
    )
    while True:
        decoy_arms = tuple(
            sorted(rng.choice(n_arms, size=action_size, replace=False))
        )
        if decoy_arms != good_arms:
            break

    rewards = rng.uniform(0.0, 1.0 - advantage, (horizon, n_arms))
    rewards[:, good_arms] = np.clip(
        rewards[:, good_arms] + advantage, 0.0, 1.0
    )
    corrupted = rng.random(horizon) < corruption_probability
    rewards[np.ix_(corrupted, decoy_arms)] = 1.0
    return rewards


def geometric_blocks(
    n_arms: int,
    action_size: int,
    horizon: int,
    seed: int | None = None,
    advantage: float = 0.1,
    n_blocks: int = DEFAULT_N_BLOCKS,
) -> np.ndarray:
    """Return rewards whose good arms change over geometric blocks.

    Block lengths are proportional to ``1, 2, 4, ...`` and sum exactly to
    ``horizon``. Every block uses a distinct random set of good arms.
    """
    _validate_parameters(n_arms, action_size, horizon, advantage)
    if not 1 <= n_blocks <= horizon:
        raise ValueError("n_blocks must be between 1 and horizon")
    if n_blocks > comb(n_arms, action_size):
        raise ValueError("n_blocks exceeds the number of distinct subsets")

    rng = np.random.default_rng(seed)
    rewards = rng.uniform(0.0, 1.0 - advantage, (horizon, n_arms))
    used_subsets: set[tuple[int, ...]] = set()
    start = 0

    for block_size in _geometric_block_sizes(horizon, n_blocks):
        while True:
            good_arms = tuple(
                sorted(rng.choice(n_arms, size=action_size, replace=False))
            )
            if good_arms not in used_subsets:
                used_subsets.add(good_arms)
                break

        stop = start + block_size
        rewards[start:stop, good_arms] = np.clip(
            rewards[start:stop, good_arms] + advantage, 0.0, 1.0
        )
        start = stop

    return rewards


def _geometric_block_sizes(horizon: int, n_blocks: int) -> np.ndarray:
    weights = np.exp2(np.arange(n_blocks, dtype=float))
    exact_sizes = horizon * weights / weights.sum()
    sizes = np.maximum(np.floor(exact_sizes).astype(int), 1)

    while sizes.sum() > horizon:
        largest = int(np.argmax(np.where(sizes > 1, sizes, -1)))
        sizes[largest] -= 1

    remainder = horizon - int(sizes.sum())
    if remainder:
        fractions = exact_sizes - np.floor(exact_sizes)
        order = np.argsort(-fractions, kind="stable")
        sizes[order[:remainder]] += 1
    return sizes


__all__ = [
    "DEFAULT_BERNOULLI_N_GOOD_ARMS",
    "DEFAULT_CORRELATED_N_GOOD_ARMS",
    "bernoulli",
    "correlated",
    "corrupted_stationary_good_arms",
    "geometric_blocks",
    "stationary_good_arms",
]
