import unittest

import numpy as np

from konbini.algorithms.osma import (
    _dependent_round,
    _estimate_losses,
    _project,
    default_learning_rate,
    osma,
)
from konbini.envs.semi_bandit import SemiBandit


class TestOSMA(unittest.TestCase):
    def test_default_learning_rate_and_none_are_equivalent(self):
        rewards = np.tile(np.array([0.8, 0.6, 0.3, 0.1]), (20, 1))
        expected = np.sqrt(2 * 2 * np.log(4 * 2) / (4 * 20))
        self.assertAlmostEqual(default_learning_rate(4, 2, 20), expected)

        default_result = osma(SemiBandit(4, 2, rewards), None, seed=7)
        explicit_result = osma(
            SemiBandit(4, 2, rewards), expected, seed=7
        )
        for key in default_result:
            np.testing.assert_array_equal(
                default_result[key], explicit_result[key]
            )

    def test_sampled_actions_are_feasible(self):
        rng = np.random.default_rng(4)
        marginals = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.4])

        for _ in range(1_000):
            action = _dependent_round(marginals, 3, rng)
            self.assertEqual(action.dtype, np.int8)
            self.assertTrue(np.all((action == 0) | (action == 1)))
            self.assertEqual(action.sum(), 3)

    def test_dependent_rounding_preserves_marginals_empirically(self):
        rng = np.random.default_rng(12)
        marginals = np.array([0.05, 0.25, 0.6, 0.8, 0.3])
        samples = np.array(
            [_dependent_round(marginals, 2, rng) for _ in range(30_000)]
        )

        np.testing.assert_allclose(samples.mean(axis=0), marginals, atol=0.012)

    def test_kl_projection_is_feasible(self):
        log_weights = np.array(
            [-800.0, -20.0, -4.0, -1.0, 0.0, 3.0, 7.0, 25.0, 500.0]
        )
        projected = _project(log_weights, 4)

        self.assertTrue(np.all(projected >= 0.0))
        self.assertTrue(np.all(projected <= 1.0))
        self.assertAlmostEqual(projected.sum(), 4.0, places=12)

    def test_kl_projection_does_not_overshoot_a_nearly_capped_coordinate(self):
        log_weights = np.array([
            -3.489658225625257,
            -6.372294012325019,
            -0.0018410410944995682,
            -5.833010287894091,
            -4.613571532901332,
            -1.3275756513689694,
            -9.983873720305232,
            0.0,
            -0.0061217981057057605,
            -0.3599489195596695,
        ])

        projected = _project(log_weights, 4)

        self.assertTrue(np.all(projected >= 0.0))
        self.assertTrue(np.all(projected <= 1.0))
        self.assertAlmostEqual(projected.sum(), 4.0, places=12)

    def test_importance_weighted_estimator_is_unbiased(self):
        rng = np.random.default_rng(33)
        marginals = np.array([0.1, 0.3, 0.45, 0.55, 0.6])
        rewards = np.array([0.05, 0.25, 0.5, 0.8, 1.0])
        estimate_sum = np.zeros(5)
        repetitions = 60_000

        for _ in range(repetitions):
            action = _dependent_round(marginals, 2, rng)
            feedback = action * rewards
            estimate_sum += _estimate_losses(action, feedback, marginals, 2)

        np.testing.assert_allclose(
            estimate_sum / repetitions,
            1.0 - rewards,
            atol=0.025,
        )

    def test_fixed_seeds_reproduce_trajectories(self):
        rewards = np.tile(np.array([0.9, 0.7, 0.4, 0.1]), (100, 1))
        results = [
            osma(SemiBandit(4, 2, rewards), learning_rate=0.08, seed=91)
            for _ in range(2)
        ]

        for key in results[0]:
            np.testing.assert_array_equal(results[0][key], results[1][key])

    def test_learns_highest_reward_coordinates(self):
        rewards_per_arm = np.array([0.95, 0.85, 0.35, 0.2, 0.1, 0.05])
        horizon = 3_000
        environment = SemiBandit(
            6, 2, np.tile(rewards_per_arm, (horizon, 1))
        )
        selections = osma(
            environment, learning_rate=0.03, seed=8
        )["actions"]

        early_best_rate = selections[:50, :2].mean()
        late_best_rate = selections[-300:, :2].mean()
        self.assertGreater(late_best_rate, early_best_rate + 0.3)
        self.assertGreater(late_best_rate, 0.85)


if __name__ == "__main__":
    unittest.main()
