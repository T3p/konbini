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
    PLOTS_DIR,
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
    plot_dir: Path | None = None,
    plot_stem: str | None = None,
    publication_style: str | None = None,
    axis_width: float | None = None,
    axis_height: float | None = None,
    write_png: bool = True,
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
    plot_output_stem = plot_stem or PLOT_OUTPUT_STEMS.get(
        file_stem, f"{file_stem}_regret"
    )
    destination = output_dir if plot_dir is None else plot_dir
    if publication_style is None:
        publication_style = (
            "main"
            if plot_output_stem
            in {"correlated", "action_size", "corrupted_main",
                "correlated_full_regret"}
            else "appendix"
        )
    plot_path = destination / f"{plot_output_stem}.png"
    if write_png:
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
            double_column=publication_style == "appendix",
        )
    tex_path = destination / f"{plot_output_stem}.tex"
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
        publication_style=publication_style,
        axis_width=axis_width,
        axis_height=axis_height,
        marker_repeat=1 if by_action_size else 100,
    )
    pdf_path = _compile_tikz(tex_path)
    return plot_path, tex_path, pdf_path


def _compile_tikz(tex_path: Path) -> Path:
    compiler = shutil.which("pdflatex")
    if compiler is None:
        raise RuntimeError("pdflatex is required to compile publication plots")
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


PUBLICATION_PLOTS = (
    ("correlated", "correlated", "correlated", False, "main", 0.95),
    (
        "correlated", "correlated_by_action_size", "action_size",
        True, "main", 0.95,
    ),
    (
        "corrupted_stationary_good_arms",
        "corrupted_stationary_good_arms",
        "corrupted_main", False, "main", 0.95,
    ),
    ("correlated", "correlated_full", "full", False, "appendix", 0.99),
    (
        "stationary_good_arms", "stationary_good_arms", "uniform",
        False, "appendix", 0.95,
    ),
    (
        "corrupted_stationary_good_arms",
        "corrupted_stationary_good_arms",
        "corrupted", False, "appendix", 0.95,
    ),
    (
        "geometric_blocks", "geometric_blocks", "blocks",
        False, "appendix", 0.95,
    ),
    ("bernoulli", "bernoulli", "bernoulli", False, "appendix", 0.95),
)

PUBLICATION_AXIS_DIMENSIONS = {
    "correlated": (2.5006, 1.6491),
    "action_size": (2.5213, 1.6506),
    "corrupted_main": (2.6182, 1.5377),
    "full": (5.9843, 2.9402),
    "uniform": (5.9843, 2.9402),
    "corrupted": (6.1123, 2.7769),
    "blocks": (6.1123, 2.7769),
    "bernoulli": (6.1123, 2.7769),
    "correlated_full_regret": (2.5006, 1.6491),
}


def build_publication_plots(
    data_dir: Path = RESULTS_DIR,
    plot_dir: Path = PLOTS_DIR,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 0,
    include_correlated_full_regret: bool = False,
) -> list[tuple[Path, Path]]:
    """Build the paper and appendix plots from the saved experiment CSVs."""
    jobs = list(PUBLICATION_PLOTS)
    if include_correlated_full_regret:
        jobs.append(
            (
                "correlated", "correlated_full", "correlated_full_regret",
                False, "main", 0.99,
            )
        )
    artifacts = []
    for instance, data_stem, plot_stem, by_action_size, style, level in jobs:
        axis_width, axis_height = PUBLICATION_AXIS_DIMENSIONS[plot_stem]
        _, tex_path, pdf_path = plot_instance(
            instance,
            data_dir,
            output_stem=data_stem,
            by_action_size=by_action_size,
            n_bootstrap=n_bootstrap,
            confidence_level=level,
            bootstrap_seed=bootstrap_seed,
            plot_dir=plot_dir,
            plot_stem=plot_stem,
            publication_style=style,
            axis_width=axis_width,
            axis_height=axis_height,
            write_png=False,
        )
        artifacts.append((tex_path, pdf_path))
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild comparison plots from existing CSV files."
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        choices=INSTANCE_NAMES,
        default=None,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=RESULTS_DIR
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PLOTS_DIR
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
    parser.add_argument(
        "--correlated-full-regret",
        action="store_true",
        help="also build the separate main-paper correlated_full_regret plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.instances is None:
        artifacts = build_publication_plots(
            args.data_dir,
            args.output_dir,
            n_bootstrap=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            include_correlated_full_regret=args.correlated_full_regret,
        )
        for tex_path, pdf_path in artifacts:
            print(f"TikZ plot: {tex_path}")
            print(f"PDF plot: {pdf_path}")
        return
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
                args.data_dir,
                output_stem=output_stem,
                by_action_size=by_action_size,
                n_bootstrap=args.bootstrap_samples,
                confidence_level=confidence_level,
                bootstrap_seed=args.bootstrap_seed,
                plot_dir=args.output_dir,
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
