"""Shared configuration for the comparison scripts."""

from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

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
    "corrupted_stationary_good_arms",
    "geometric_blocks",
)


def csv_path(output_dir: Path, instance: str, algorithm: str) -> Path:
    return output_dir / f"{instance}_{CSV_FILENAMES[algorithm]}"
