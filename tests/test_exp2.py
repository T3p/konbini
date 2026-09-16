import unittest

import numpy as np

from konbini.algorithms.exp2 import (
    _reward_update,
    default_parameters,
    enumerate_actions,
    exp2,
    uniform_exploration_moment,
)
from konbini.envs.full_bandit import FullBandit


class AverageFullBandit(FullBandit):
    def step(self, action, render=False):
        observation, reward, terminated, truncated, info = super().step(
            action, render
        )
        return (
            observation / self.action_size,
            reward / self.action_size,
            terminated,
            truncated,
            info,
        )


class TestEXP2(unittest.TestCase):
    def test_analytic_exploration_moment_matches_enumeration(self):
        actions = enumerate_actions(6, 3)
        enumerated = actions.T @ actions / len(actions)

        np.testing.assert_allclose(
            uniform_exploration_moment(6, 3), enumerated, atol=1e-15
        )

    def test_mixed_moment_matches_direct_sampling_distribution(self):
        actions = enumerate_actions(5, 2).astype(float)
        exploitation = np.arange(1, len(actions) + 1, dtype=float)
        exploitation /= exploitation.sum()
        gamma = 0.3
        sampling = (1 - gamma) * exploitation + gamma / len(actions)

        direct = (actions.T * sampling) @ actions
        analytic = (
            (1 - gamma) * (actions.T * exploitation) @ actions
            + gamma * uniform_exploration_moment(5, 2)
        )
        np.testing.assert_allclose(analytic, direct, atol=1e-15)

    def test_action_reward_estimator_is_exactly_unbiased(self):
        actions = enumerate_actions(4, 2).astype(float)
        exploitation = np.arange(1, len(actions) + 1, dtype=float)
        exploitation /= exploitation.sum()
        gamma = 0.2
        sampling = (1 - gamma) * exploitation + gamma / len(actions)
        moment = (actions.T * sampling) @ actions
        coordinate_rewards = np.array([0.1, 0.3, 0.6, 0.8])
        action_rewards = actions @ coordinate_rewards

        expected = sum(
            sampling[index]
            * actions
            @ np.linalg.solve(
                moment, actions[index] * action_rewards[index]
            )
            for index in range(len(actions))
        )
        np.testing.assert_allclose(expected, action_rewards, atol=1e-12)

    def test_sum_and_average_updates_are_equivalent(self):
        n_arms, action_size, horizon = 5, 2, 100
        rewards = np.random.default_rng(8).random((horizon, n_arms))
        sum_result = exp2(
            FullBandit(n_arms, action_size, rewards), seed=13
        )
        average_result = exp2(
            AverageFullBandit(n_arms, action_size, rewards),
            seed=13,
            reward_scale="average",
        )

        np.testing.assert_array_equal(
            sum_result["actions"], average_result["actions"]
        )
        np.testing.assert_allclose(
            sum_result["rewards"] / action_size,
            average_result["rewards"],
        )

    def test_default_parameters_and_seeded_run(self):
        eta, gamma = default_parameters(5, 2, 100)
        self.assertLessEqual(gamma, 0.5)
        self.assertAlmostEqual(eta * 2 * 5 / gamma, 1.0)

        rewards = np.random.default_rng(2).random((100, 5))
        results = [
            exp2(FullBandit(5, 2, rewards), seed=21) for _ in range(2)
        ]
        np.testing.assert_array_equal(
            results[0]["actions"], results[1]["actions"]
        )
        self.assertTrue(np.all(results[0]["actions"].sum(axis=1) == 2))

    def test_positive_reward_update_uses_ascent(self):
        updated = _reward_update(
            np.zeros(3), np.array([0.0, 1.0, -0.5]), eta=0.2
        )
        self.assertGreater(updated[1], updated[0])
        self.assertLess(updated[2], updated[0])

    def test_single_action_skips_learning(self):
        rewards = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float
        )
        result = exp2(FullBandit(3, 3, rewards), seed=4)

        np.testing.assert_array_equal(
            result["actions"], np.ones((2, 3), dtype=np.int8)
        )
        np.testing.assert_allclose(result["rewards"], [0.6, 1.5])


if __name__ == "__main__":
    unittest.main()
