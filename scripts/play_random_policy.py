"""Run a random fixed-cardinality policy in one of the bandit environments.

Example:
    python play_random_policy.py --env semi-bandit --horizon 20
"""

import argparse
from pathlib import Path

import numpy as np

from konbini.envs.full_bandit import FullBandit
from konbini.envs.instances import stationary_good_arms
from konbini.envs.multiplayer_mab import MultiplayerMab
from konbini.envs.semi_bandit import SemiBandit
from konbini.envs.win_win import WinWin
from regret_plot import save_regret_plot


# Change these values to customize runs without passing CLI arguments.
DEFAULT_ENV = "full-bandit"
DEFAULT_N_ARMS = 5
DEFAULT_ACTION_SIZE = 2
DEFAULT_HORIZON = 20000
DEFAULT_SEED = 0
DEFAULT_WINNER = True

ENVIRONMENTS = {
    "full-bandit": FullBandit,
    "semi-bandit": SemiBandit,
    "multiplayer-mab": MultiplayerMab,
    "win-win": WinWin,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a random policy in a combinatorial bandit environment."
    )
    parser.add_argument(
        "--env",
        choices=ENVIRONMENTS,
        default=DEFAULT_ENV,
        help=f"environment to run (default: {DEFAULT_ENV})",
    )
    parser.add_argument("--n-arms", type=int, default=DEFAULT_N_ARMS)
    parser.add_argument("--action-size", type=int, default=DEFAULT_ACTION_SIZE)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--winner",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WINNER,
        help=(
            "enable winner feedback for full-bandit "
            f"(default: {DEFAULT_WINNER})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reward_seed, policy_seed = np.random.SeedSequence(args.seed).spawn(2)
    rewards = stationary_good_arms(
        args.n_arms,
        args.action_size,
        args.horizon,
        seed=int(reward_seed.generate_state(1)[0]),
    )

    env_class = ENVIRONMENTS[args.env]
    env_options = dict(
        n_arms=args.n_arms,
        action_size=args.action_size,
        rewards=rewards,
        horizon=args.horizon,
    )
    if env_class is FullBandit:
        env_options["winner_feedback"] = args.winner
    env = env_class(**env_options)
    policy_rng = np.random.default_rng(policy_seed)

    observation, _ = env.reset(seed=args.seed)
    total_reward = 0.0
    actions = []

    for timestep in range(1, args.horizon + 1):
        print(f"timestep {timestep}, rewards: ", end="")
        env.render()

        selected_arms = policy_rng.choice(
            args.n_arms, size=args.action_size, replace=False
        )
        action = np.zeros(args.n_arms, dtype=np.int8)
        action[selected_arms] = 1

        observation, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        total_reward += reward
        print(
            f"  selected={selected_arms.tolist()}, reward={reward:.4f}, "
            f"observation={np.asarray(observation).tolist()}"
        )
        print(f"  info={info}")

        if terminated or truncated:
            break

    actions = np.asarray(actions)
    played_rewards = rewards[:len(actions)]
    plot_path = Path(__file__).parent / "results" / "random_policy_regret.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    best_subset = save_regret_plot(
        played_rewards,
        {"Random policy": actions},
        args.action_size,
        plot_path,
        "Random policy regret",
    )
    env.close()
    print(f"best fixed action in hindsight: {list(best_subset)}")
    print(f"total reward: {total_reward:.4f}")
    print(f"regret plot: {plot_path}")


if __name__ == "__main__":
    main()
