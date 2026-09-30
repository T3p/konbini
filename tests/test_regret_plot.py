import unittest

import numpy as np

from scripts.regret_plot import blockwise_cumulative_regret


class TestRegretPlot(unittest.TestCase):
    def test_blockwise_regret_uses_each_blocks_best_fixed_subset(self):
        rewards = np.array(
            [
                [0.8, 0.2, 0.1],
                [0.3, 0.7, 0.1],
                [0.6, 0.4, 0.1],
                [0.2, 0.9, 0.1],
            ]
        )
        actions = np.tile(np.array([0, 0, 1]), (4, 1))

        regret = blockwise_cumulative_regret(
            rewards, actions, action_size=1, block_sizes=np.array([2, 2])
        )

        np.testing.assert_allclose(regret, np.array([0.7, 0.9, 1.2, 2.0]))


if __name__ == "__main__":
    unittest.main()
