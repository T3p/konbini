"""Rebuild comparison plots from saved regret CSV files."""

import argparse
from pathlib import Path
import shutil
import subprocess

import numpy as np

from comparison_config import (
    ALGORITHM_LABELS,
    INSTANCE_NAMES,
    PLOT_LABELS,
    PLOT_OUTPUT_STEMS,
    RESULTS_DIR,
    csv_path,
)
from regret_plot import save_bootstrap_regret_plot, save_tikz_regret_plot


def plot_instance(
    instance: str,
    output_dir: Path = RESULTS_DIR,
    output_stem: str | None = None,
    by_action_size: bool = False,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 0,
) -> tuple[Path, Path, Path]:
    """Rebuild one instance plot from its saved CSV files."""
    if instance not in INSTANCE_NAMES:
        raise ValueError(f"unknown instance: {instance}")
    file_stem = instance if output_stem is None else output_stem
    regrets_by_label = {}
    x_values = None
    arm_counts = None
    for algorithm in ALGORITHM_LABELS:
        result_path = csv_path(output_dir, file_stem, algorithm)
        if not result_path.is_file():
            continue
        table = np.loadtxt(
            result_path,
            delimiter=",",
            skiprows=1,
            ndmin=2,
        )
        if by_action_size:
            header = result_path.read_text(encoding="utf-8").splitlines()[0]
            if header.split(",")[:2] != ["action_size", "n_arms"]:
                raise ValueError(
                    f"{result_path.name} predates the fixed-arm experiment; "
                    "regenerate the action-size logs"
                )
            current_x_values = table[:, 0]
            current_arm_counts = table[:, 1]
            if x_values is None:
                x_values = current_x_values
                arm_counts = current_arm_counts
            elif not np.array_equal(x_values, current_x_values):
                raise ValueError(
                    "action sizes must match across algorithm CSV files"
                )
            elif not np.array_equal(arm_counts, current_arm_counts):
                raise ValueError(
                    "arm counts must match across algorithm CSV files"
                )
            if (
                np.any(~np.isfinite(current_arm_counts))
                or np.any(current_arm_counts <= 0)
                or np.any(current_arm_counts != np.floor(current_arm_counts))
            ):
                raise ValueError(
                    "arm counts must be positive integers"
                )
            regrets = table[:, 2:]
        else:
            regrets = table[:, 1:]
        regrets_by_label[PLOT_LABELS[algorithm]] = regrets
    if not regrets_by_label:
        raise FileNotFoundError(
            f"no regret CSV files found in {output_dir}"
        )

    y_label = (
        "Final total regret"
        if by_action_size
        else (
            "Total blockwise regret"
            if instance == "geometric_blocks"
            else "Total regret"
        )
    )
    double_column = instance == "correlated"
    plot_output_stem = PLOT_OUTPUT_STEMS.get(
        file_stem, f"{file_stem}_regret"
    )
    plot_path = output_dir / f"{plot_output_stem}.png"
    save_bootstrap_regret_plot(
        regrets_by_label,
        plot_path,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
        y_label=y_label,
        x_values=x_values,
        x_label="Action size" if by_action_size else "Rounds",
        show_all_x_ticks=by_action_size,
        include_x_origin=not by_action_size,
        double_column=double_column,
    )
    tex_path = output_dir / f"{plot_output_stem}.tex"
    save_tikz_regret_plot(
        regrets_by_label,
        tex_path,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
        y_label=y_label,
        x_values=x_values,
        x_label="Action size" if by_action_size else "Rounds",
        show_all_x_ticks=by_action_size,
        include_x_origin=not by_action_size,
        double_column=double_column,
    )
    pdf_path = _compile_tikz(tex_path)
    return plot_path, tex_path, pdf_path


def _compile_tikz(tex_path: Path) -> Path:
    user_tectonic = Path.home() / ".local" / "bin" / "tectonic"
    compiler = (
        shutil.which("pdflatex")
        or shutil.which("lualatex")
        or shutil.which("tectonic")
        or (str(user_tectonic) if user_tectonic.is_file() else None)
    )
    if compiler is None:
        raise RuntimeError(
            "a TeX compiler is required; install pdflatex, lualatex, or tectonic"
        )
    if Path(compiler).name == "tectonic":
        command = [
            compiler,
            "--outdir",
            str(tex_path.parent),
            str(tex_path),
        ]
    else:
        command = [
            compiler,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={tex_path.parent}",
            str(tex_path),
        ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"failed to compile {tex_path.name}:\n{completed.stdout}\n"
            f"{completed.stderr}"
        )
    for suffix in (".aux", ".log"):
        tex_path.with_suffix(suffix).unlink(missing_ok=True)
    return tex_path.with_suffix(".pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild comparison plots from existing CSV files."
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        choices=INSTANCE_NAMES,
        default=list(INSTANCE_NAMES),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS_DIR
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--full",
        action="store_true",
        help="rebuild plots from full-mode CSV files",
    )
    parser.add_argument(
        "--m",
        action="store_true",
        help="rebuild action-size comparison plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_jobs = []
    for instance in args.instances:
        output_stem = instance
        if args.m:
            output_stem += "_by_action_size"
        if args.full:
            output_stem += "_full"
        plot_jobs.append(
            (
                instance,
                output_stem,
                args.m,
                0.99 if args.full else args.confidence_level,
            )
        )
    if (
        not args.full
        and not args.m
        and "stationary_good_arms" in args.instances
    ):
        plot_jobs.append(
            (
                "stationary_good_arms",
                "stationary_good_arms_full",
                False,
                0.99,
            )
        )

    plotted_instances = 0
    for (
        instance,
        output_stem,
        by_action_size,
        confidence_level,
    ) in plot_jobs:
        try:
            png_path, tex_path, pdf_path = plot_instance(
                instance,
                args.output_dir,
                output_stem=output_stem,
                by_action_size=by_action_size,
                n_bootstrap=args.bootstrap_samples,
                confidence_level=confidence_level,
                bootstrap_seed=args.bootstrap_seed,
            )
        except FileNotFoundError as error:
            print(f"{instance}: skipped ({error})")
            continue
        plotted_instances += 1
        print(f"{instance} PNG plot: {png_path}")
        print(f"{instance} TikZ plot: {tex_path}")
        print(f"{instance} PDF plot: {pdf_path}")
    if not plotted_instances:
        print("No plots generated: no regret CSV files were found.")


if __name__ == "__main__":
    main()
