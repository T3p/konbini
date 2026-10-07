"""Shared configuration for the comparison scripts."""

from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent.parent / "plots"

ALGORITHM_LABELS = ("OSMA", "OSMA-W", "EXP2", "K-Metaplayer")
CSV_FILENAMES = {
    "OSMA": "osma_regret.csv",
    "OSMA-W": "osmaw_regret.csv",
    "EXP2": "exp2_regret.csv",
    "K-Metaplayer": "k_metaplayer_regret.csv",
}
PLOT_LABELS = {
    "OSMA": "OSMA on semi-bandit",
    "OSMA-W": "OSMA-W on FB+WF",
    "EXP2": "EXP2 on full-bandit",
    "K-Metaplayer": "K-Metaplayer on random-arm",
}
INSTANCE_NAMES = (
    "stationary_good_arms",
    "bernoulli",
    "correlated",
    "corrupted_stationary_good_arms",
    "geometric_blocks",
)
PLOT_OUTPUT_STEMS = {
    "bernoulli": "bernoulli",
    "correlated_by_action_size": "action_size",
    "correlated": "correlated",
    "correlated_full": "full",
    "corrupted_stationary_good_arms_by_action_size": (
        "action_size_corrupted"
    ),
    "corrupted_stationary_good_arms": "corrupted",
    "geometric_blocks": "blocks",
    "stationary_good_arms_by_action_size": "action_size_uniform",
    "stationary_good_arms_full": "uniform_full",
    "stationary_good_arms": "uniform",
}


def csv_path(output_dir: Path, instance: str, algorithm: str) -> Path:
    return output_dir / f"{instance}_{CSV_FILENAMES[algorithm]}"
