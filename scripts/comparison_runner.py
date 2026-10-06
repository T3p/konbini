"""Shared execution code for instance-specific comparison scripts."""

import argparse
from collections.abc import Callable
from math import comb
from pathlib import Path

import numpy as np

from comparison_config import (
    ALGORITHM_LABELS,
    CSV_FILENAMES,
    RESULTS_DIR,
    csv_path,
)
from konbini.algorithms.exp2 import exp2
from konbini.algorithms.k_metaplayer import k_metaplayer
from konbini.algorithms.osma import osma
from konbini.algorithms.osmaw import osmaw
from konbini.envs.full_bandit import FullBandit
from konbini.envs.instances import _geometric_block_sizes
from konbini.envs.multiplayer_mab import MultiplayerMab
from konbini.envs.semi_bandit import SemiBandit
from regret_plot import (
    best_fixed_subset,
    blockwise_cumulative_regret,
    cumulative_regret,
    save_regret_csv,
    seed_column_names,
)

def build_parser(
    description: str,
    default_n_arms: int,
    default_action_size: int,
    default_horizon: int | None,
    default_reward_seed: int | None,
    default_algorithm_seeds: tuple[int, ...] | None,
) -> argparse.ArgumentParser:
    """Build a parser from defaults owned by an experiment script."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--n-arms", type=int, default=default_n_arms)
    parser.add_argument(
        "--action-size", type=int, default=default_action_size
    )
    parser.add_argument("--horizon", type=int, default=default_horizon)
    parser.add_argument(
        "--seed",
        type=int,
        default=default_reward_seed,
        help="seed used once to generate the shared reward matrix",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=(
            None
            if default_algorithm_seeds is None
            else list(default_algorithm_seeds)
        ),
        help="random seeds used for every algorithm",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="directory for the CSV files and plot",
    )
    return parser


def run_experiment(
    instance: str,
    reward_factory: Callable[..., np.ndarray],
    args: argparse.Namespace,
    blockwise_n_blocks: int | None = None,
    independent_rewards: bool = False,
    output_stem: str | None = None,
) -> None:
    """Run all algorithms on one reward instance and save their regrets."""
    block_sizes = (
        _geometric_block_sizes(args.horizon, blockwise_n_blocks)
        if blockwise_n_blocks is not None
        else None
    )
    shared_rewards = (
        None
        if independent_rewards
        else reward_factory(
            args.n_arms,
            args.action_size,
            args.horizon,
            seed=args.seed,
        )
    )
    shared_best_subset = (
        best_fixed_subset(shared_rewards, args.action_size)
        if shared_rewards is not None and block_sizes is None
        else None
    )
    regret_runs_by_label: dict[str, list[np.ndarray]] = {
        label: [] for label in ALGORITHM_LABELS
    }

    algorithm_seeds = tuple(args.seeds)
    print(f"algorithm seeds: {list(algorithm_seeds)}")
    for run_number, seed in enumerate(algorithm_seeds, start=1):
        rewards = (
            reward_factory(
                args.n_arms,
                args.action_size,
                args.horizon,
                seed=seed,
            )
            if independent_rewards
            else shared_rewards
        )
        assert rewards is not None
        best_subset = (
            best_fixed_subset(rewards, args.action_size)
            if block_sizes is None and independent_rewards
            else shared_best_subset
        )
        print(
            f"{instance}: running seed {seed} "
            f"({run_number}/{len(algorithm_seeds)})"
        )
        for label, actions in _run_algorithms(
            rewards, args.action_size, seed
        ).items():
            if block_sizes is not None:
                regret = blockwise_cumulative_regret(
                    rewards, actions, args.action_size, block_sizes
                )
            else:
                assert best_subset is not None
                regret = cumulative_regret(rewards, actions, best_subset)
            regret_runs_by_label[label].append(regret)

    regrets_by_label = {
        label: np.column_stack(regret_runs)
        for label, regret_runs in regret_runs_by_label.items()
    }

    file_stem = instance if output_stem is None else output_stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, regrets in regrets_by_label.items():
        output_path = csv_path(args.output_dir, file_stem, label)
        save_regret_csv(regrets, algorithm_seeds, output_path)
        print(f"{label} regret CSV: {output_path}")

    from plot_comparison import plot_instance

    png_path, tex_path, pdf_path = plot_instance(
        instance,
        args.output_dir,
        output_stem=file_stem,
    )
    if blockwise_n_blocks is not None:
        print("comparator: best fixed subset within each block")
    elif independent_rewards:
        print("comparator: best fixed action for each reward matrix")
    else:
        assert shared_best_subset is not None
        print(
            "best fixed action in hindsight: "
            f"{list(shared_best_subset)}"
        )
    for label, regrets in regrets_by_label.items():
        print(f"{label} mean final regret: {regrets[-1].mean():.4f}")
    print(f"PNG plot: {png_path}")
    print(f"TikZ plot: {tex_path}")
    print(f"PDF plot: {pdf_path}")


def run_action_size_experiment(
    instance: str,
    reward_factory: Callable[..., np.ndarray],
    args: argparse.Namespace,
    action_sizes: tuple[int, ...],
    n_arms: int,
    blockwise_n_blocks: int | None = None,
    independent_rewards: bool = False,
    output_stem: str | None = None,
) -> None:
    """Compare final regret across action sizes with a fixed arm count."""
    if (
        not action_sizes
        or tuple(sorted(set(action_sizes))) != action_sizes
        or any(not 1 <= size < n_arms for size in action_sizes)
    ):
        raise ValueError(
            "action_sizes must be strictly increasing values between "
            "1 and n_arms - 1"
        )
    algorithm_seeds = tuple(args.seeds)
    final_regrets_by_label = {
        label: np.empty((len(action_sizes), len(algorithm_seeds)))
        for label in ALGORITHM_LABELS
    }
    print(f"algorithm seeds: {list(algorithm_seeds)}")

    for size_index, action_size in enumerate(action_sizes):
        size = int(action_size)
        block_sizes = (
            _geometric_block_sizes(args.horizon, blockwise_n_blocks)
            if blockwise_n_blocks is not None
            else None
        )
        shared_rewards = (
            None
            if independent_rewards
            else reward_factory(
                n_arms, size, args.horizon, seed=args.seed
            )
        )
        shared_best_subset = (
            best_fixed_subset(shared_rewards, size)
            if shared_rewards is not None and block_sizes is None
            else None
        )

        for run_index, seed in enumerate(algorithm_seeds):
            rewards = (
                reward_factory(n_arms, size, args.horizon, seed=seed)
                if independent_rewards
                else shared_rewards
            )
            assert rewards is not None
            best_subset = (
                best_fixed_subset(rewards, size)
                if block_sizes is None and independent_rewards
                else shared_best_subset
            )
            print(
                f"{instance}, action size {size}/{action_sizes[-1]}, "
                f"arms {n_arms}: seed {seed} "
                f"({run_index + 1}/{len(algorithm_seeds)})"
            )
            actions_by_label = _run_algorithms(
                rewards,
                size,
                seed,
                exp2_max_actions=comb(n_arms, size),
            )
            for label, actions in actions_by_label.items():
                if block_sizes is not None:
                    regret = blockwise_cumulative_regret(
                        rewards, actions, size, block_sizes
                    )
                else:
                    assert best_subset is not None
                    regret = cumulative_regret(
                        rewards, actions, best_subset
                    )
                final_regrets_by_label[label][size_index, run_index] = (
                    regret[-1]
                )

    file_stem = (
        f"{instance}_by_action_size"
        if output_stem is None
        else output_stem
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, final_regrets in final_regrets_by_label.items():
        output_path = (
            args.output_dir / f"{file_stem}_{CSV_FILENAMES[label]}"
        )
        _save_action_size_regret_csv(
            action_sizes,
            n_arms,
            final_regrets,
            algorithm_seeds,
            output_path,
        )
        print(f"{label} final-regret CSV: {output_path}")

    from plot_comparison import plot_instance

    png_path, tex_path, pdf_path = plot_instance(
        instance,
        args.output_dir,
        output_stem=file_stem,
        by_action_size=True,
    )
    comparator = (
        "best fixed subset within each block"
        if blockwise_n_blocks is not None
        else "best fixed action in hindsight"
    )
    print(f"comparator: {comparator}")
    for label, final_regrets in final_regrets_by_label.items():
        values = ", ".join(
            f"m={size}: {mean:.4f}"
            for size, mean in zip(
                action_sizes, final_regrets.mean(axis=1)
            )
        )
        print(f"{label} mean final regret: {values}")
    print(f"PNG plot: {png_path}")
    print(f"TikZ plot: {tex_path}")
    print(f"PDF plot: {pdf_path}")


def _save_action_size_regret_csv(
    action_sizes: tuple[int, ...],
    n_arms: int,
    final_regrets: np.ndarray,
    seeds: tuple[int, ...],
    output_path: Path,
) -> None:
    arm_counts = np.full(len(action_sizes), n_arms)
    rows = np.column_stack((action_sizes, arm_counts, final_regrets))
    header = ",".join(
        ("action_size", "n_arms", *seed_column_names(seeds))
    )
    np.savetxt(
        output_path,
        rows,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", "%d", *(["%.18g"] * len(seeds))],
    )


def _run_algorithms(
    rewards: np.ndarray,
    action_size: int,
    seed: int,
    exp2_max_actions: int = 100_000,
) -> dict[str, np.ndarray]:
    horizon, n_arms = rewards.shape
    environments = {
        "OSMA": SemiBandit(n_arms, action_size, rewards, horizon),
        "OSMA-W": FullBandit(
            n_arms, action_size, rewards, horizon, winner_feedback=True
        ),
        "EXP2": FullBandit(
            n_arms, action_size, rewards, horizon, winner_feedback=False
        ),
        "K-Metaplayer": MultiplayerMab(
            n_arms, action_size, rewards, horizon
        ),
    }
    try:
        results = {
            "OSMA": osma(
                environments["OSMA"], learning_rate=None, seed=seed
            ),
            "OSMA-W": osmaw(
                environments["OSMA-W"],
                learning_rate=None,
                exploration_coefficient=None,
                seed=seed,
            ),
            "EXP2": exp2(
                environments["EXP2"],
                eta=None,
                gamma=None,
                seed=seed,
                max_actions=exp2_max_actions,
            ),
            "K-Metaplayer": k_metaplayer(
                environments["K-Metaplayer"], eta=None, seed=seed
            ),
        }
    finally:
        for environment in environments.values():
            environment.close()
    return {label: results[label]["actions"] for label in ALGORITHM_LABELS}
