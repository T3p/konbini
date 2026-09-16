"""Negative-entropy online stochastic mirror descent for semi-bandits."""

from __future__ import annotations

import numpy as np


def osma(environment, learning_rate: float, seed: int | None = None) -> dict:
    """Run negative-entropy OSMD for one semi-bandit episode.

    The returned dictionary contains the sampled ``actions`` and scalar
    ``rewards``.
    """
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    n_arms = environment.n_arms
    action_size = environment.action_size
    rng = np.random.default_rng(seed)
    marginals = np.full(n_arms, action_size / n_arms, dtype=float)

    actions = []
    rewards = []
    environment.reset(seed=seed)

    while True:
        pre_action_marginals = marginals.copy()
        action = _dependent_round(pre_action_marginals, action_size, rng)
        feedback, reward, terminated, truncated, _ = environment.step(action)
        loss_estimate = _estimate_losses(
            action, feedback, pre_action_marginals, action_size
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


def _dependent_round(
    marginals: np.ndarray,
    action_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    probabilities = _validate_marginals(marginals, action_size).copy()
    residual = action_size - float(probabilities.sum())
    if residual:
        adjustable = np.flatnonzero(
            probabilities < 1.0 if residual > 0 else probabilities > 0.0
        )
        probabilities[adjustable[0]] += residual

    while True:
        fractional = np.flatnonzero(
            (probabilities > 0.0) & (probabilities < 1.0)
        )
        if fractional.size < 2:
            break

        first, second = fractional[:2]
        first_value = probabilities[first]
        second_value = probabilities[second]
        increase = min(1.0 - first_value, second_value)
        decrease = min(first_value, 1.0 - second_value)

        if rng.random() < decrease / (increase + decrease):
            probabilities[first] = first_value + increase
            probabilities[second] = second_value - increase
            if increase == 1.0 - first_value:
                probabilities[first] = 1.0
            if increase == second_value:
                probabilities[second] = 0.0
        else:
            probabilities[first] = first_value - decrease
            probabilities[second] = second_value + decrease
            if decrease == first_value:
                probabilities[first] = 0.0
            if decrease == 1.0 - second_value:
                probabilities[second] = 1.0

    fractional = np.flatnonzero(
        (probabilities > 0.0) & (probabilities < 1.0)
    )
    if fractional.size == 1:
        index = fractional[0]
        distance = min(probabilities[index], 1.0 - probabilities[index])
        if distance <= 1e-12:
            probabilities[index] = float(probabilities[index] >= 0.5)
        else:
            raise RuntimeError("dependent rounding left a fractional coordinate")
    elif fractional.size:
        raise RuntimeError("dependent rounding left fractional coordinates")

    action = probabilities.astype(np.int8)
    if int(action.sum()) != action_size:
        raise RuntimeError("dependent rounding produced an infeasible action")
    return action


def _estimate_losses(
    action: np.ndarray,
    feedback: np.ndarray,
    sampling_marginals: np.ndarray,
    action_size: int,
    bounded_rewards: bool = True,
) -> np.ndarray:
    action = np.asarray(action)
    feedback = np.asarray(feedback, dtype=float)
    sampling_marginals = np.asarray(sampling_marginals, dtype=float)
    if action.shape != feedback.shape or action.shape != sampling_marginals.shape:
        raise ValueError("action, feedback, and marginals must have equal shapes")
    if (np.any((action != 0) & (action != 1))
            or int(action.sum()) != action_size):
        raise ValueError("action must be a feasible binary subset")

    selected = action.astype(bool)
    selected_rewards = feedback[selected]
    if not np.all(np.isfinite(selected_rewards)):
        raise ValueError("selected coordinate rewards must be finite")
    if (bounded_rewards
            and np.any((selected_rewards < 0) | (selected_rewards > 1))):
        raise ValueError("selected coordinate rewards must lie in [0, 1]")

    denominators = sampling_marginals[selected]
    if not np.all(np.isfinite(denominators)) or np.any(denominators <= 0):
        raise ZeroDivisionError("selected sampling marginals must be positive")

    estimate = np.zeros(action.size, dtype=float)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            estimate[selected] = (1.0 - selected_rewards) / denominators
        except FloatingPointError as error:
            raise FloatingPointError(
                "loss estimate is not representable at the supplied marginals"
            ) from error
    return estimate


def _project(log_weights: np.ndarray, action_size: int) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("log_weights must be a finite vector")

    log_floor = np.log(np.nextafter(0.0, 1.0))
    with np.errstate(over="ignore"):
        shifted = values - np.max(values)
    shifted = np.maximum(shifted, log_floor)
    lower = float(np.min(shifted))
    upper = float(-np.log(action_size / values.size))

    for _ in range(100):
        multiplier = (lower + upper) / 2.0
        projected = np.exp(np.minimum(0.0, shifted - multiplier))
        if projected.sum() > action_size:
            lower = multiplier
        else:
            upper = multiplier

    projected = np.exp(
        np.minimum(0.0, shifted - (lower + upper) / 2.0)
    )
    projected = np.maximum(projected, np.nextafter(0.0, 1.0))
    residual = action_size - float(projected.sum())
    if residual:
        adjustable = np.flatnonzero(
            projected < 1.0 if residual > 0 else projected > 0.0
        )
        projected[adjustable[np.argmax(projected[adjustable])]] += residual

    if (np.any(projected < 0.0) or np.any(projected > 1.0)
            or not np.isclose(projected.sum(), action_size, atol=1e-12)):
        raise RuntimeError("KL projection failed")
    return projected


def _validate_marginals(
    marginals: np.ndarray,
    action_size: int,
) -> np.ndarray:
    values = np.asarray(marginals, dtype=float)
    if (values.ndim != 1
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
            or not np.isclose(
                values.sum(), action_size, atol=1e-10, rtol=0.0
            )):
        raise ValueError(
            "marginals must lie in [0, 1] and sum to action_size"
        )
    return values
