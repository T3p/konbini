"""EXP2 with uniform exploration for fixed-cardinality subsets.

All ``binomial(d, m)`` actions are explicitly enumerated. Uniform exploration
has a closed-form second moment, but time and memory still grow exponentially
with the number of base arms when ``m`` is not near zero or ``d``.
"""

from __future__ import annotations

from itertools import combinations
from math import comb, lgamma

import numpy as np


def enumerate_actions(
    n_arms: int,
    action_size: int,
    max_actions: int = 100_000,
) -> np.ndarray:
    """Enumerate every binary ``action_size``-subset."""
    if n_arms < 1 or not 1 <= action_size <= n_arms:
        raise ValueError("action_size must be between 1 and n_arms")
    action_count = comb(n_arms, action_size)
    if action_count > max_actions:
        raise ValueError(
            f"action set has {action_count} elements; max_actions={max_actions}"
        )

    actions = np.zeros((action_count, n_arms), dtype=np.int8)
    for index, subset in enumerate(combinations(range(n_arms), action_size)):
        actions[index, subset] = 1
    return actions


def default_parameters(
    n_arms: int,
    action_size: int,
    horizon: int,
    reward_scale: str = "sum",
) -> tuple[float, float]:
    """Return the horizon-based ``(eta, gamma)`` defaults."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    _validate_reward_scale(reward_scale)
    log_action_count = (
        lgamma(n_arms + 1)
        - lgamma(action_size + 1)
        - lgamma(n_arms - action_size + 1)
    )
    gamma = min(0.5, (n_arms * log_action_count / (3 * horizon)) ** 0.5)
    denominator = n_arms if reward_scale == "average" else action_size * n_arms
    return gamma / denominator, gamma


def exp2(
    environment,
    eta: float | None = None,
    gamma: float | None = None,
    seed: int | None = None,
    reward_scale: str = "sum",
    max_actions: int = 100_000,
) -> dict:
    """Run EXP2 for one full-bandit episode.

    ``reward_scale`` explicitly declares whether the scalar observation is the
    selected reward sum in ``[0, m]`` or its average in ``[0, 1]``. Defaults
    are derived from ``environment.horizon``; either may be overridden.
    """
    n_arms = environment.n_arms
    action_size = environment.action_size
    _validate_reward_scale(reward_scale)

    if action_size == n_arms:
        return _play_single_action(environment, seed, reward_scale)

    default_eta, default_gamma = default_parameters(
        n_arms, action_size, environment.horizon, reward_scale
    )
    eta = default_eta if eta is None else eta
    gamma = default_gamma if gamma is None else gamma
    _validate_parameters(eta, gamma, n_arms, action_size, reward_scale)

    actions = enumerate_actions(n_arms, action_size, max_actions)
    action_count = actions.shape[0]
    exploration_moment = uniform_exploration_moment(n_arms, action_size)
    rng = np.random.default_rng(seed)
    log_weights = np.zeros(action_count, dtype=float)
    played_actions = []
    rewards = []
    environment.reset(seed=seed)

    while True:
        exploitation = _softmax(log_weights)
        sampling = (
            (1.0 - gamma) * exploitation + gamma / action_count
        )
        action_index = int(rng.choice(action_count, p=sampling))
        action = actions[action_index]
        observation, reward, terminated, truncated, _ = environment.step(
            action
        )
        observed_reward = _read_reward(
            observation, action_size, reward_scale
        )

        exploitation_moment = (actions.T * exploitation) @ actions
        moment = (
            (1.0 - gamma) * exploitation_moment
            + gamma * exploration_moment
        )
        reward_estimate = np.linalg.solve(
            moment, action * observed_reward
        )
        action_reward_estimates = actions @ reward_estimate
        log_weights = _reward_update(
            log_weights, action_reward_estimates, eta
        )
        played_actions.append(action.copy())
        rewards.append(reward)

        if terminated or truncated:
            break

    return {
        "actions": np.asarray(played_actions, dtype=np.int8),
        "rewards": np.asarray(rewards, dtype=float),
    }


def uniform_exploration_moment(
    n_arms: int,
    action_size: int,
) -> np.ndarray:
    """Return the analytic second moment of a uniform random m-subset."""
    diagonal = action_size / n_arms
    off_diagonal = (
        action_size * (action_size - 1) / (n_arms * (n_arms - 1))
    )
    moment = np.full((n_arms, n_arms), off_diagonal, dtype=float)
    np.fill_diagonal(moment, diagonal)
    return moment


def _softmax(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    return weights / weights.sum()


def _reward_update(
    log_weights: np.ndarray,
    estimated_rewards: np.ndarray,
    eta: float,
) -> np.ndarray:
    with np.errstate(over="raise", invalid="raise"):
        try:
            updated = log_weights + eta * estimated_rewards
        except FloatingPointError as error:
            raise FloatingPointError("EXP2 reward update overflowed") from error
    return updated - np.max(updated)


def _read_reward(
    observation: np.ndarray,
    action_size: int,
    reward_scale: str,
) -> float:
    observed = np.asarray(observation, dtype=float)
    if observed.size != 1 or not np.all(np.isfinite(observed)):
        raise ValueError("full-bandit observation must be a finite scalar")
    reward = float(observed.item())
    upper_bound = 1.0 if reward_scale == "average" else float(action_size)
    if not 0 <= reward <= upper_bound:
        raise ValueError(
            f"{reward_scale} reward must lie in [0, {upper_bound:g}]"
        )
    return reward


def _play_single_action(environment, seed: int | None, reward_scale: str) -> dict:
    action = np.ones(environment.n_arms, dtype=np.int8)
    actions = []
    rewards = []
    environment.reset(seed=seed)
    while True:
        observation, reward, terminated, truncated, _ = environment.step(
            action
        )
        _read_reward(observation, environment.action_size, reward_scale)
        actions.append(action.copy())
        rewards.append(reward)
        if terminated or truncated:
            break
    return {
        "actions": np.asarray(actions, dtype=np.int8),
        "rewards": np.asarray(rewards, dtype=float),
    }


def _validate_reward_scale(reward_scale: str) -> None:
    if reward_scale not in {"sum", "average"}:
        raise ValueError("reward_scale must be 'sum' or 'average'")


def _validate_parameters(
    eta: float,
    gamma: float,
    n_arms: int,
    action_size: int,
    reward_scale: str,
) -> None:
    if not np.isfinite(gamma) or not 0 < gamma <= 0.5:
        raise ValueError("gamma must lie in (0, 1/2]")
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and positive")
    reward_factor = 1 if reward_scale == "average" else action_size
    if eta * reward_factor * n_arms / gamma > 1.0 + 1e-12:
        raise ValueError(
            "theory condition violated: eta * reward_factor * d / gamma > 1"
        )
