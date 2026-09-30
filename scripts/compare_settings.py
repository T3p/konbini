"""Compare all algorithms on one selectable reward instance."""

import argparse
from collections.abc import Callable
from functools import partial

import numpy as np

from comparison_runner import (
    build_parser,
    run_action_size_experiment,
    run_experiment,
)
from konbini.envs.instances import (
    corrupted_stationary_good_arms,
    geometric_blocks,
    stationary_good_arms,
)


DEFAULT_N_ARMS = 10
DEFAULT_ACTION_SIZE = 4
DEFAULT_HORIZON = 50_000
DEFAULT_ADVANTAGE = 0.1
DEFAULT_CORRUPTION_PROBABILITY = 0.2
DEFAULT_N_BLOCKS = 4

STATIONARY_REWARD_SEED = 4594
CORRUPTED_REWARD_SEED = 2385
BLOCKING_REWARD_SEED = 3849
STANDARD_ALGORITHM_SEEDS = (58, 26, 36, 50, 23, 65, 93, 49, 85, 86)
BLOCKING_ALGORITHM_SEEDS = (18, 76)


def add_instance_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared reward-instance selection options."""
    instance_group = parser.add_mutually_exclusive_group()
    instance_group.add_argument(
        "--corrupted",
        action="store_true",
        help="use stationary good arms with corrupted decoy rounds",
    )
    instance_group.add_argument(
        "--blocking",
        action="store_true",
        help="use good arms that change over geometric blocks",
    )
    parser.add_argument("--advantage", type=float, default=DEFAULT_ADVANTAGE)
    parser.add_argument(
        "--corruption-probability",
        type=float,
        default=DEFAULT_CORRUPTION_PROBABILITY,
    )
    parser.add_argument("--n-blocks", type=int, default=DEFAULT_N_BLOCKS)


def select_instance(
    args: argparse.Namespace,
) -> tuple[
    str,
    Callable[..., np.ndarray],
    int,
    tuple[int, ...],
    int | None,
]:
    """Return the selected instance and its mode-specific defaults."""
    if args.corrupted:
        instance = "corrupted_stationary_good_arms"
        reward_factory = partial(
            corrupted_stationary_good_arms,
            advantage=args.advantage,
            corruption_probability=args.corruption_probability,
        )
        default_reward_seed = CORRUPTED_REWARD_SEED
        default_algorithm_seeds = STANDARD_ALGORITHM_SEEDS
        blockwise_n_blocks = None
    elif args.blocking:
        instance = "geometric_blocks"
        reward_factory = partial(
            geometric_blocks,
            advantage=args.advantage,
            n_blocks=args.n_blocks,
        )
        default_reward_seed = BLOCKING_REWARD_SEED
        default_algorithm_seeds = BLOCKING_ALGORITHM_SEEDS
        blockwise_n_blocks = args.n_blocks
    else:
        instance = "stationary_good_arms"
        reward_factory = partial(
            stationary_good_arms,
            advantage=args.advantage,
        )
        default_reward_seed = STATIONARY_REWARD_SEED
        default_algorithm_seeds = STANDARD_ALGORITHM_SEEDS
        blockwise_n_blocks = None

    return (
        instance,
        reward_factory,
        default_reward_seed,
        default_algorithm_seeds,
        blockwise_n_blocks,
    )


def main() -> None:
    parser = build_parser(
        "Compare feedback models on a selectable reward instance.",
        DEFAULT_N_ARMS,
        DEFAULT_ACTION_SIZE,
        None,
        None,
        None,
    )
    add_instance_arguments(parser)
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "run independent reward matrices for 100 seeds "
            "(100 through 199 by default)"
        ),
    )
    parser.add_argument(
        "--m",
        action="store_true",
        help=(
            "compare action sizes 2 through 10 with twice as many arms"
        ),
    )
    args = parser.parse_args()
    (
        instance,
        reward_factory,
        default_reward_seed,
        default_algorithm_seeds,
        blockwise_n_blocks,
    ) = select_instance(args)

    if args.horizon is None:
        args.horizon = 10_000 if args.m else DEFAULT_HORIZON
    if args.seed is None:
        args.seed = default_reward_seed
    if args.seeds is None:
        args.seeds = (
            list(range(100, 200))
            if args.full
            else (
                list(STANDARD_ALGORITHM_SEEDS[:5])
                if args.m
                else list(default_algorithm_seeds)
            )
        )

    if args.m:
        output_stem = f"{instance}_by_action_size"
        if args.full:
            output_stem += "_full"
        run_action_size_experiment(
            instance,
            reward_factory,
            args,
            blockwise_n_blocks=blockwise_n_blocks,
            independent_rewards=args.full,
            output_stem=output_stem,
        )
    else:
        run_experiment(
            instance,
            reward_factory,
            args,
            blockwise_n_blocks=blockwise_n_blocks,
            independent_rewards=args.full,
            output_stem=(f"{instance}_full" if args.full else instance),
        )


if __name__ == "__main__":
    main()
