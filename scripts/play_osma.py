"""Compare OSMA, OSMA-W, and EXP2 on a shared reward matrix.

Example:
    python scripts/play_osma.py --n-arms 5 --action-size 2 --horizon 20
"""

import argparse
from pathlib import Path

import numpy as np

from konbini.algorithms.exp2 import default_parameters, exp2
from konbini.algorithms.osma import osma
from konbini.algorithms.osmaw import osmaw
from konbini.envs.full_bandit import FullBandit
from konbini.envs.semi_bandit import SemiBandit
from regret_plot import generate_reward_matrix, save_regret_plot


DEFAULT_N_ARMS = 5
DEFAULT_ACTION_SIZE = 2
DEFAULT_HORIZON = 40000
DEFAULT_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OSMA, OSMA-W, and EXP2 on shared rewards."
    )
    parser.add_argument("--n-arms", type=int, default=DEFAULT_N_ARMS)
    parser.add_argument("--action-size", type=int, default=DEFAULT_ACTION_SIZE)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reward_seed, osma_seed, osmaw_seed, exp2_seed = np.random.SeedSequence(
        args.seed
    ).spawn(4)
    rewards = generate_reward_matrix(
        args.n_arms,
        args.action_size,
        args.horizon,
        np.random.default_rng(reward_seed),
    )
    semi_bandit = SemiBandit(
        n_arms=args.n_arms,
        action_size=args.action_size,
        rewards=rewards,
        horizon=args.horizon,
    )
    full_bandit = FullBandit(
        n_arms=args.n_arms,
        action_size=args.action_size,
        rewards=rewards,
        horizon=args.horizon,
        winner_feedback=True,
    )
    exp2_bandit = FullBandit(
        n_arms=args.n_arms,
        action_size=args.action_size,
        rewards=rewards,
        horizon=args.horizon,
        winner_feedback=False,
    )
    osma_learning_rate = np.sqrt(
        2
        * args.action_size
        * np.log(args.n_arms * args.action_size)
        / (args.n_arms * args.horizon)
    )
    osmaw_learning_rate = min(
        np.sqrt(
            args.action_size * np.log(args.n_arms / args.action_size)
            / (2 * args.n_arms * args.horizon)
        ),
        (3 - np.e) / args.n_arms,
    )
    exploration_coefficient = osmaw_learning_rate * args.n_arms
    exp2_eta, exp2_gamma = default_parameters(
        args.n_arms,
        args.action_size,
        args.horizon,
    )

    osma_result = osma(
        semi_bandit,
        learning_rate=osma_learning_rate,
        seed=int(osma_seed.generate_state(1)[0]),
    )
    osmaw_result = osmaw(
        full_bandit,
        learning_rate=osmaw_learning_rate,
        exploration_coefficient=exploration_coefficient,
        seed=int(osmaw_seed.generate_state(1)[0]),
    )
    exp2_result = exp2(
        exp2_bandit,
        seed=int(exp2_seed.generate_state(1)[0]),
    )

    rounds = zip(
        osma_result["actions"],
        osma_result["rewards"],
        osmaw_result["actions"],
        osmaw_result["rewards"],
        exp2_result["actions"],
        exp2_result["rewards"],
    )
    for timestep, values in enumerate(rounds, start=1):
        (
            osma_action,
            osma_reward,
            osmaw_action,
            osmaw_reward,
            exp2_action,
            exp2_reward,
        ) = values
        print(f"timestep {timestep}, rewards: {rewards[timestep - 1]}")
        print(
            f"  OSMA selected={np.flatnonzero(osma_action).tolist()}, "
            f"reward={osma_reward:.4f}"
        )
        print(
            f"  OSMA-W selected={np.flatnonzero(osmaw_action).tolist()}, "
            f"reward={osmaw_reward:.4f}"
        )
        print(
            f"  EXP2 selected={np.flatnonzero(exp2_action).tolist()}, "
            f"reward={exp2_reward:.4f}"
        )

    plot_path = Path(__file__).with_name("osma_regret.png")
    best_subset = save_regret_plot(
        rewards,
        {
            "OSMA": osma_result["actions"],
            "OSMA-W": osmaw_result["actions"],
            "EXP2": exp2_result["actions"],
        },
        args.action_size,
        plot_path,
        "OSMA, OSMA-W, and EXP2 regret",
    )
    semi_bandit.close()
    full_bandit.close()
    exp2_bandit.close()
    print(f"best fixed action in hindsight: {list(best_subset)}")
    print(f"OSMA learning rate: {osma_learning_rate:.6f}")
    print(f"OSMA-W learning rate: {osmaw_learning_rate:.6f}")
    print(f"exploration coefficient: {exploration_coefficient:.6f}")
    print(f"EXP2 eta: {exp2_eta:.6f}")
    print(f"EXP2 gamma: {exp2_gamma:.6f}")
    print(f"OSMA total reward: {osma_result['rewards'].sum():.4f}")
    print(f"OSMA-W total reward: {osmaw_result['rewards'].sum():.4f}")
    print(f"EXP2 total reward: {exp2_result['rewards'].sum():.4f}")
    print(f"regret plot: {plot_path}")


if __name__ == "__main__":
    main()
