"""Centralized K-Metaplayer with direct reward updates.

The paper analyzes negative loss updates. This module instead performs a
positive reward update, so the paper's stated regret guarantee does not
automatically transfer. Sampling and marginal computation take ``O(d * m)``
time and memory per round and do not enumerate the subset action space.
"""

from __future__ import annotations

from math import lgamma

import numpy as np


def default_learning_rate(n_arms: int, action_size: int, horizon: int) -> float:
    """Return ``sqrt(log(C(K, m)) / (m * K * T))``."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    log_action_count = (
        lgamma(n_arms + 1)
        - lgamma(action_size + 1)
        - lgamma(n_arms - action_size + 1)
    )
    denominator = action_size * n_arms * horizon
    return np.sqrt(log_action_count / denominator)


def k_metaplayer(
    environment,
    eta: float | None = None,
    seed: int | None = None,
) -> dict:
    """Run the centralized direct-reward K-Metaplayer for one episode."""
    n_arms = environment.n_arms
    action_size = environment.action_size
    if action_size == n_arms:
        return _play_single_action(environment, seed)

    if eta is None:
        eta = default_learning_rate(n_arms, action_size, environment.horizon)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and positive")

    rng = np.random.default_rng(seed)
    log_weights = np.zeros(n_arms, dtype=float)
    actions = []
    rewards = []
    environment.reset(seed=seed)

    while True:
        prefix, suffix = _log_elementary_tables(log_weights, action_size)
        marginals = _inclusion_marginals(
            log_weights, action_size, prefix, suffix
        )
        action = _sample_subset(log_weights, action_size, suffix, rng)
        observation, reward, terminated, truncated, info = environment.step(
            action
        )

        if "winner" not in info:
            raise ValueError("K-Metaplayer requires single-arm feedback")
        observed_arm = int(info["winner"])
        if observed_arm < 0 or observed_arm >= n_arms or not action[observed_arm]:
            raise ValueError("observed arm must belong to the sampled subset")
        observed = np.asarray(observation, dtype=float)
        if (observed.size != 1
                or not np.all(np.isfinite(observed))
                or not 0 <= observed.item() <= 1):
            raise ValueError("observed coordinate reward must lie in [0, 1]")
        denominator = marginals[observed_arm]
        if not np.isfinite(denominator) or denominator <= 0:
            raise ZeroDivisionError("observed-arm marginal must be positive")

        reward_estimate = action_size * float(observed.item()) / denominator
        if not np.isfinite(reward_estimate):
            raise FloatingPointError("reward estimate is not finite")
        log_weights[observed_arm] += eta * reward_estimate
        if not np.all(np.isfinite(log_weights)):
            raise FloatingPointError("K-Metaplayer update overflowed")
        log_weights -= np.max(log_weights)

        actions.append(action)
        rewards.append(reward)
        if terminated or truncated:
            break

    return {
        "actions": np.asarray(actions, dtype=np.int8),
        "rewards": np.asarray(rewards, dtype=float),
    }


def _log_elementary_tables(
    log_weights: np.ndarray,
    action_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_arms = log_weights.size
    prefix = np.full((n_arms + 1, action_size + 1), -np.inf)
    suffix = np.full((n_arms + 1, action_size + 1), -np.inf)
    prefix[0, 0] = 0.0
    suffix[n_arms, 0] = 0.0

    for arm in range(n_arms):
        prefix[arm + 1, 0] = 0.0
        for count in range(1, min(action_size, arm + 1) + 1):
            prefix[arm + 1, count] = np.logaddexp(
                prefix[arm, count],
                log_weights[arm] + prefix[arm, count - 1],
            )

    for arm in range(n_arms - 1, -1, -1):
        suffix[arm, 0] = 0.0
        available = n_arms - arm
        for count in range(1, min(action_size, available) + 1):
            suffix[arm, count] = np.logaddexp(
                suffix[arm + 1, count],
                log_weights[arm] + suffix[arm + 1, count - 1],
            )
    return prefix, suffix


def _inclusion_marginals(
    log_weights: np.ndarray,
    action_size: int,
    prefix: np.ndarray,
    suffix: np.ndarray,
) -> np.ndarray:
    n_arms = log_weights.size
    log_partition = suffix[0, action_size]
    if not np.isfinite(log_partition):
        raise FloatingPointError("subset partition function is not finite")

    marginals = np.empty(n_arms, dtype=float)
    for arm in range(n_arms):
        terms = []
        for left_count in range(action_size):
            right_count = action_size - 1 - left_count
            if left_count <= arm and right_count <= n_arms - arm - 1:
                terms.append(
                    prefix[arm, left_count]
                    + suffix[arm + 1, right_count]
                )
        log_without_arm = np.logaddexp.reduce(terms)
        marginals[arm] = np.exp(
            log_weights[arm] + log_without_arm - log_partition
        )

    roundoff_tolerance = 1e-9
    if (not np.all(np.isfinite(marginals))
            or np.any(marginals < -roundoff_tolerance)
            or np.any(marginals > 1.0 + roundoff_tolerance)):
        raise FloatingPointError("invalid inclusion marginals")
    marginals = np.clip(marginals, 0.0, 1.0)
    if not np.isclose(
        marginals.sum(), action_size, atol=roundoff_tolerance, rtol=0.0
    ):
        raise FloatingPointError("invalid inclusion marginals")
    return marginals


def _sample_subset(
    log_weights: np.ndarray,
    action_size: int,
    suffix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n_arms = log_weights.size
    action = np.zeros(n_arms, dtype=np.int8)
    remaining = action_size

    for arm in range(n_arms):
        arms_left = n_arms - arm
        if remaining == 0:
            break
        if remaining == arms_left:
            action[arm:] = 1
            remaining = 0
            break

        log_probability = (
            log_weights[arm]
            + suffix[arm + 1, remaining - 1]
            - suffix[arm, remaining]
        )
        if not np.isfinite(log_probability) or log_probability > 1e-12:
            raise FloatingPointError("invalid conditional inclusion probability")
        probability = np.exp(min(0.0, log_probability))
        if rng.random() < probability:
            action[arm] = 1
            remaining -= 1

    if remaining != 0 or int(action.sum()) != action_size:
        raise RuntimeError("subset sampler produced an infeasible action")
    return action


def _play_single_action(environment, seed: int | None) -> dict:
    action = np.ones(environment.n_arms, dtype=np.int8)
    actions = []
    rewards = []
    environment.reset(seed=seed)
    while True:
        _, reward, terminated, truncated, _ = environment.step(action)
        actions.append(action.copy())
        rewards.append(reward)
        if terminated or truncated:
            break
    return {
        "actions": np.asarray(actions, dtype=np.int8),
        "rewards": np.asarray(rewards, dtype=float),
    }
