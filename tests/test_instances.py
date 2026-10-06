import unittest

import numpy as np

from konbini.envs.instances import (
    DEFAULT_N_BLOCKS,
    _geometric_block_sizes,
    bernoulli,
    correlated,
    corrupted_stationary_good_arms,
    geometric_blocks,
    stationary_good_arms,
)


class TestInstances(unittest.TestCase):
    def test_bernoulli_is_seeded_binary_and_uses_random_parameters(self):
        n_arms = 12
        horizon = 100_000
        advantage = 0.1
        n_good_arms = 5
        seed = 41
        rewards = bernoulli(
            n_arms,
            2,
            horizon,
            seed=seed,
            advantage=advantage,
            n_good_arms=n_good_arms,
        )
        repeated = bernoulli(
            n_arms,
            2,
            horizon,
            seed=seed,
            advantage=advantage,
            n_good_arms=n_good_arms,
        )

        rng = np.random.default_rng(seed)
        expected_parameters = rng.uniform(
            0.0, 1.0 - advantage, n_arms
        )
        good_arms = rng.choice(
            n_arms, size=n_good_arms, replace=False
        )
        expected_parameters[good_arms] += advantage

        np.testing.assert_array_equal(rewards, repeated)
        self.assertTrue(np.all((rewards == 0.0) | (rewards == 1.0)))
        np.testing.assert_allclose(
            rewards.mean(axis=0), expected_parameters, atol=0.01
        )

    def test_bernoulli_uses_same_instance_across_action_sizes(self):
        small_action = bernoulli(
            20, 2, 1_000, seed=43, n_good_arms=10
        )
        large_action = bernoulli(
            20, 10, 1_000, seed=43, n_good_arms=10
        )

        np.testing.assert_array_equal(small_action, large_action)

    def test_bernoulli_validates_good_arm_count(self):
        with self.assertRaises(ValueError):
            bernoulli(6, 2, 10, seed=0, n_good_arms=1.5)
        with self.assertRaises(ValueError):
            bernoulli(6, 2, 10, seed=0, n_good_arms=6)

    def test_correlated_is_seeded_bounded_and_has_shared_round_noise(self):
        rewards = correlated(16, 2, 10_000, seed=29)
        repeated = correlated(16, 2, 10_000, seed=29)

        np.testing.assert_array_equal(rewards, repeated)
        self.assertTrue(np.all((rewards >= 0.25) & (rewards <= 0.75)))
        row_ranges = np.ptp(rewards, axis=1)
        self.assertTrue(
            np.all(np.isclose(row_ranges, 0.0) | np.isclose(row_ranges, 0.25))
        )

    def test_correlated_has_requested_mean_advantage(self):
        advantage = 0.1
        action_size = 4
        n_good_arms = 10
        rewards = correlated(
            20,
            action_size,
            100_000,
            seed=31,
            advantage=advantage,
            n_good_arms=n_good_arms,
        )
        means = np.sort(rewards.mean(axis=0))

        self.assertAlmostEqual(
            means[-n_good_arms:].mean() - means[:-n_good_arms].mean(),
            advantage,
            delta=0.01,
        )

    def test_correlated_uses_same_instance_across_action_sizes(self):
        small_action = correlated(20, 2, 1_000, seed=37, n_good_arms=10)
        large_action = correlated(20, 10, 1_000, seed=37, n_good_arms=10)

        np.testing.assert_array_equal(small_action, large_action)

    def test_correlated_validates_bernoulli_probability(self):
        correlated(
            20, 2, 10, seed=0, advantage=3.0 / 16.0, n_good_arms=10
        )
        with self.assertRaises(ValueError):
            correlated(
                20, 2, 10, seed=0, advantage=0.19, n_good_arms=10
            )
        with self.assertRaises(ValueError):
            correlated(20, 2, 10, seed=0, n_good_arms=20)

    def test_corrupted_stationary_good_arms(self):
        horizon = 20_000
        rewards = corrupted_stationary_good_arms(
            6, 2, horizon, seed=21, corruption_probability=0.2
        )
        repeated = corrupted_stationary_good_arms(
            6, 2, horizon, seed=21, corruption_probability=0.2
        )

        np.testing.assert_array_equal(rewards, repeated)
        self.assertTrue(np.all((rewards >= 0.0) & (rewards <= 1.0)))

        decoy_rewards = rewards == 1.0
        corrupted = np.any(decoy_rewards, axis=1)
        self.assertAlmostEqual(corrupted.mean(), 0.2, delta=0.015)
        self.assertTrue(np.all(decoy_rewards[corrupted].sum(axis=1) == 2))
        self.assertEqual(np.count_nonzero(np.any(decoy_rewards, axis=0)), 2)

    def test_corruption_probability_extremes_and_validation(self):
        clean = corrupted_stationary_good_arms(
            5, 2, 20, seed=3, corruption_probability=0.0
        )
        corrupted = corrupted_stationary_good_arms(
            5, 2, 20, seed=3, corruption_probability=1.0
        )

        self.assertFalse(np.any(clean == 1.0))
        self.assertTrue(np.all(np.sum(corrupted == 1.0, axis=1) == 2))
        for invalid in (-0.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                corrupted_stationary_good_arms(
                    5, 2, 20, seed=3, corruption_probability=invalid
                )

    def test_geometric_block_sizes(self):
        np.testing.assert_array_equal(
            _geometric_block_sizes(310, 5),
            np.array([10, 20, 40, 80, 160]),
        )
        sizes = _geometric_block_sizes(8, 5)
        self.assertEqual(sizes.sum(), 8)
        self.assertTrue(np.all(sizes >= 1))
        self.assertTrue(np.all(np.diff(sizes) >= 0))

    def test_geometric_blocks_are_seeded_bounded_and_distinct(self):
        horizon = 3_100
        advantage = 0.99
        rewards = geometric_blocks(
            6, 2, horizon, seed=13, advantage=advantage
        )
        repeated = geometric_blocks(
            6, 2, horizon, seed=13, advantage=advantage
        )

        np.testing.assert_array_equal(rewards, repeated)
        self.assertTrue(np.all((rewards >= 0.0) & (rewards <= 1.0)))

        good_sets = []
        start = 0
        for size in _geometric_block_sizes(horizon, DEFAULT_N_BLOCKS):
            stop = start + size
            good = np.all(rewards[start:stop] >= advantage, axis=0)
            good_sets.append(tuple(np.flatnonzero(good)))
            start = stop

        self.assertTrue(all(len(good_set) == 2 for good_set in good_sets))
        self.assertEqual(len(set(good_sets)), DEFAULT_N_BLOCKS)

    def test_geometric_blocks_validates_block_count(self):
        with self.assertRaises(ValueError):
            geometric_blocks(4, 2, 4, seed=0, n_blocks=5)
        with self.assertRaises(ValueError):
            geometric_blocks(4, 2, 10, seed=0, n_blocks=7)

    def test_stationary_good_arms_is_seeded_and_bounded(self):
        first = stationary_good_arms(6, 2, 100, seed=17)
        second = stationary_good_arms(6, 2, 100, seed=17)

        self.assertEqual(first.shape, (100, 6))
        self.assertTrue(np.all((first >= 0.0) & (first <= 1.0)))
        np.testing.assert_array_equal(first, second)

    def test_stationary_good_arms_has_requested_mean_advantage(self):
        advantage = 0.2
        rewards = stationary_good_arms(
            6, 2, 100_000, seed=4, advantage=advantage
        )
        means = np.sort(rewards.mean(axis=0))
        self.assertAlmostEqual(
            means[-2:].mean() - means[:-2].mean(),
            advantage,
            delta=0.01,
        )

    def test_stationary_good_arms_validates_parameters(self):
        with self.assertRaises(ValueError):
            stationary_good_arms(4, 4, 10, seed=0)
        with self.assertRaises(ValueError):
            stationary_good_arms(4, 2, 0, seed=0)
        with self.assertRaises(ValueError):
            stationary_good_arms(4, 2, 10, seed=0, advantage=1.0)


if __name__ == "__main__":
    unittest.main()
