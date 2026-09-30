import importlib
import unittest
from unittest.mock import patch

import numpy as np

from konbini.envs.full_bandit import FullBandit


class TestOSMAW(unittest.TestCase):
    def test_theoretical_defaults_accept_none(self):
        module = importlib.import_module("konbini.algorithms.osmaw")
        n_arms = 4
        action_size = 2
        horizon = 20
        expected_rate = min(
            np.sqrt(
                action_size * np.log(n_arms / action_size)
                / (2 * n_arms * horizon)
            ),
            (3 - np.e) / n_arms,
        )
        self.assertAlmostEqual(
            module.default_learning_rate(n_arms, action_size, horizon),
            expected_rate,
        )
        self.assertAlmostEqual(
            module.default_exploration_coefficient(
                n_arms, action_size, horizon
            ),
            n_arms * expected_rate,
        )

        rewards = np.tile(np.array([0.8, 0.6, 0.3, 0.1]), (horizon, 1))
        default_result = module.osmaw(
            FullBandit(4, 2, rewards, winner_feedback=True),
            learning_rate=None,
            exploration_coefficient=None,
            seed=7,
        )
        explicit_result = module.osmaw(
            FullBandit(4, 2, rewards, winner_feedback=True),
            learning_rate=expected_rate,
            exploration_coefficient=n_arms * expected_rate,
            seed=7,
        )
        for key in default_result:
            np.testing.assert_array_equal(
                default_result[key], explicit_result[key]
            )

    def test_recovered_rewards_are_empirically_unbiased(self):
        module = importlib.import_module("konbini.algorithms.osmaw")
        base_rewards = np.array([0.2, 0.6, 0.4, 0.9])
        action = np.array([1, 1, 1, 0], dtype=np.int8)
        horizon = 20_000
        environment = FullBandit(
            4,
            3,
            np.tile(base_rewards, (horizon, 1)),
            winner_feedback=True,
        )
        environment.reset(seed=15)
        proxy_sum = np.zeros(4)

        for _ in range(horizon):
            observation, _, _, _, info = environment.step(action)
            proxy_sum += module.recover_base_arm_rewards(
                float(observation.item()), info["winner"], action
            )

        np.testing.assert_allclose(
            proxy_sum / horizon,
            action * base_rewards,
            atol=0.015,
        )

    def test_exploration_coefficient_must_be_positive(self):
        module = importlib.import_module("konbini.algorithms.osmaw")
        environment = FullBandit(
            3, 1, np.ones((1, 3)), winner_feedback=True
        )

        with self.assertRaisesRegex(ValueError, "exploration_coefficient"):
            module.osmaw(environment, 0.1, 0.0, seed=0)

    def test_reward_proxies_must_not_exceed_bound(self):
        module = importlib.import_module("konbini.algorithms.osmaw")
        environment = FullBandit(
            3, 1, np.ones((1, 3)), winner_feedback=True
        )
        proxies = np.full(3, 30.1)

        with patch.object(
            module, "recover_base_arm_rewards", return_value=proxies
        ):
            with self.assertRaisesRegex(ValueError, "reward proxies"):
                module.osmaw(environment, 0.1, 0.1, seed=0)


if __name__ == "__main__":
    unittest.main()
