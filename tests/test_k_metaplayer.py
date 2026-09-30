import unittest
from itertools import combinations
from math import comb, log

import numpy as np

from konbini.algorithms.k_metaplayer import (
    _inclusion_marginals,
    _log_elementary_tables,
    _sample_subset,
    default_learning_rate,
    k_metaplayer,
)
from konbini.envs.multiplayer_mab import MultiplayerMab


class RecordingMultiplayerMab(MultiplayerMab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.winners = []

    def step(self, action, render=False):
        result = super().step(action, render)
        self.winners.append(result[4]["winner"])
        return result


def exact_distribution(log_weights, action_size):
    subsets = list(combinations(range(log_weights.size), action_size))
    scores = np.array([log_weights[list(subset)].sum() for subset in subsets])
    probabilities = np.exp(scores - scores.max())
    probabilities /= probabilities.sum()
    return subsets, probabilities


class TestKMetaplayer(unittest.TestCase):
    def test_default_learning_rate(self):
        n_arms = 7
        action_size = 3
        horizon = 500
        expected = np.sqrt(
            log(comb(n_arms, action_size))
            / (action_size * n_arms * horizon)
        )
        self.assertAlmostEqual(
            default_learning_rate(n_arms, action_size, horizon), expected
        )

    def test_dynamic_programming_matches_exact_enumeration(self):
        log_weights = np.array([-1.2, 0.3, 1.1, -0.4, 0.8])
        action_size = 2
        subsets, probabilities = exact_distribution(
            log_weights, action_size
        )
        prefix, suffix = _log_elementary_tables(log_weights, action_size)
        marginals = _inclusion_marginals(
            log_weights, action_size, prefix, suffix
        )
        exact_marginals = np.array(
            [
                sum(
                    probability
                    for subset, probability in zip(subsets, probabilities)
                    if arm in subset
                )
                for arm in range(log_weights.size)
            ]
        )

        np.testing.assert_allclose(
            np.exp(
                np.array(
                    [log_weights[list(subset)].sum() for subset in subsets]
                )
                - suffix[0, action_size]
            ),
            probabilities,
            atol=1e-14,
        )
        np.testing.assert_allclose(marginals, exact_marginals, atol=1e-14)

    def test_empirical_sampling_frequencies_match_marginals(self):
        log_weights = np.array([-0.7, 0.2, 1.0, -0.1, 0.5])
        action_size = 3
        prefix, suffix = _log_elementary_tables(log_weights, action_size)
        marginals = _inclusion_marginals(
            log_weights, action_size, prefix, suffix
        )
        rng = np.random.default_rng(19)
        samples = np.array(
            [
                _sample_subset(log_weights, action_size, suffix, rng)
                for _ in range(30_000)
            ]
        )

        self.assertTrue(np.all(samples.sum(axis=1) == action_size))
        np.testing.assert_allclose(samples.mean(axis=0), marginals, atol=0.012)

    def test_reward_estimator_is_unbiased(self):
        log_weights = np.array([-0.3, 0.4, 0.9, -0.1])
        action_size = 2
        prefix, suffix = _log_elementary_tables(log_weights, action_size)
        marginals = _inclusion_marginals(
            log_weights, action_size, prefix, suffix
        )
        rewards = np.array([0.1, 0.4, 0.7, 0.9])
        expectation = np.zeros(4)

        for arm in range(4):
            observation_probability = marginals[arm] / action_size
            estimate = action_size * rewards[arm] / marginals[arm]
            expectation[arm] = observation_probability * estimate

        np.testing.assert_allclose(expectation, rewards, atol=1e-15)

    def test_positive_update_favors_subsets_containing_observed_arm(self):
        before_subsets, before = exact_distribution(np.zeros(4), 2)
        updated = np.zeros(4)
        updated[1] += 0.5
        after_subsets, after = exact_distribution(updated, 2)
        self.assertEqual(before_subsets, after_subsets)

        for subset, old_probability, new_probability in zip(
            before_subsets, before, after
        ):
            if 1 in subset:
                self.assertGreater(new_probability, old_probability)
            else:
                self.assertLess(new_probability, old_probability)

    def test_seeded_gymnasium_run_is_reproducible(self):
        rewards = np.random.default_rng(2).random((100, 5))
        results = [
            k_metaplayer(MultiplayerMab(5, 2, rewards), seed=31)
            for _ in range(2)
        ]

        np.testing.assert_array_equal(
            results[0]["actions"], results[1]["actions"]
        )
        np.testing.assert_array_equal(
            results[0]["rewards"], results[1]["rewards"]
        )
        self.assertTrue(np.all(results[0]["actions"].sum(axis=1) == 2))

    def test_unobserved_rewards_do_not_affect_update(self):
        rewards = np.full((2, 5), 0.2)
        first_environment = RecordingMultiplayerMab(5, 2, rewards)
        first = k_metaplayer(first_environment, eta=0.3, seed=7)
        observed_arm = first_environment.winners[0]

        changed_rewards = rewards.copy()
        changed_rewards[0, np.arange(5) != observed_arm] = 0.9
        second_environment = RecordingMultiplayerMab(5, 2, changed_rewards)
        second = k_metaplayer(second_environment, eta=0.3, seed=7)

        np.testing.assert_array_equal(first["actions"], second["actions"])
        self.assertNotEqual(first["rewards"][0], second["rewards"][0])

    def test_single_action_skips_learning(self):
        rewards = np.array([[0.1, 0.2], [0.5, 0.4]])
        result = k_metaplayer(MultiplayerMab(2, 2, rewards), seed=4)

        np.testing.assert_array_equal(
            result["actions"], np.ones((2, 2), dtype=np.int8)
        )


if __name__ == "__main__":
    unittest.main()
