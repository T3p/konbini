"""Negative-entropy OSMD with aggregate reward and winner feedback."""

from __future__ import annotations

import numpy as np

from .osma import _dependent_round, _estimate_losses, _project


def recover_base_arm_rewards(
    observed_reward_sum: float,
    winner: int,
    action: np.ndarray,
) -> np.ndarray:
    """Construct an unbiased coordinate-reward proxy from winner feedback."""
    winner_selector = np.zeros_like(action, dtype=bool)
    winner_selector[winner] = 1
    return winner_selector * observed_reward_sum


def osmaw(
    environment,
    learning_rate: float,
    exploration_coefficient: float,
    seed: int | None = None,
) -> dict:
    """Run negative-entropy OSMD for one full-bandit episode."""
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if (not np.isfinite(exploration_coefficient)
            or not 0 < exploration_coefficient <= 1):
        raise ValueError("exploration_coefficient must lie in (0, 1]")

    n_arms = environment.n_arms
    action_size = environment.action_size
    rng = np.random.default_rng(seed)
    marginals = np.full(n_arms, action_size / n_arms, dtype=float)

    actions = []
    rewards = []
    environment.reset(seed=seed)

    while True:
        pre_action_marginals = marginals.copy()
        sampling_marginals = (
            (1.0 - exploration_coefficient) * pre_action_marginals
            + exploration_coefficient * action_size / n_arms
        )
        if rng.random() < exploration_coefficient:
            action = np.zeros(n_arms, dtype=np.int8)
            selected = rng.choice(n_arms, size=action_size, replace=False)
            action[selected] = 1
        else:
            action = _dependent_round(pre_action_marginals, action_size, rng)
        observation, reward, terminated, truncated, info = environment.step(
            action
        )

        if "winner" not in info:
            raise ValueError(
                "OSMA-W requires a full-bandit environment with winner feedback"
            )
        observed_sum = np.asarray(observation, dtype=float)
        if observed_sum.size != 1 or not np.all(np.isfinite(observed_sum)):
            raise ValueError("aggregate reward feedback must be a finite scalar")

        coordinate_rewards = np.asarray(
            recover_base_arm_rewards(
                float(observed_sum.item()),
                int(info["winner"]),
                action.copy(),
            ),
            dtype=float,
        )
        proxy_upper_bound = n_arms / exploration_coefficient
        if (coordinate_rewards.shape != action.shape
                or not np.all(np.isfinite(coordinate_rewards))
                or np.any(coordinate_rewards < 0)
                or np.any(coordinate_rewards > proxy_upper_bound)):
            raise ValueError(
                "reward proxies must have shape (n_arms,) and lie in "
                "[0, n_arms / exploration_coefficient]"
            )
        loss_estimate = _estimate_losses(
            action,
            coordinate_rewards,
            sampling_marginals,
            action_size,
            bounded_rewards=False,
        )

        with np.errstate(over="raise", invalid="raise"):
            try:
                log_weights = np.log(pre_action_marginals)
                log_weights -= learning_rate * loss_estimate
            except FloatingPointError as error:
                raise FloatingPointError(
                    "mirror update is not representable at this learning rate"
                ) from error

        marginals = _project(log_weights, action_size)
        actions.append(action)
        rewards.append(reward)

        if terminated or truncated:
            break

    return {
        "actions": np.asarray(actions, dtype=np.int8),
        "rewards": np.asarray(rewards, dtype=float),
    }
