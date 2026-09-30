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
            current_x_values = table[:, 0]
            if x_values is None:
                x_values = current_x_values
            elif not np.array_equal(x_values, current_x_values):
                raise ValueError(
                    "action sizes must match across algorithm CSV files"
                )
        regrets_by_label[PLOT_LABELS[algorithm]] = table[:, 1:]
    if not regrets_by_label:
        raise FileNotFoundError(
            f"no regret CSV files found in {output_dir}"
        )

    y_label = (
        "Final cumulative regret"
        if by_action_size
        else (
            "Cumulative blockwise regret"
            if instance == "geometric_blocks"
            else "Cumulative regret"
        )
    )
    plot_path = output_dir / f"{file_stem}_regret.png"
    save_bootstrap_regret_plot(
        regrets_by_label,
        plot_path,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
        y_label=y_label,
        x_values=x_values,
        x_label="Action size" if by_action_size else "Timestep",
        show_all_x_ticks=by_action_size,
        include_x_origin=not by_action_size,
    )
    tex_path = output_dir / f"{file_stem}_regret.tex"
    save_tikz_regret_plot(
        regrets_by_label,
        tex_path,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
        y_label=y_label,
        x_values=x_values,
        x_label="Action size" if by_action_size else "Timestep",
        show_all_x_ticks=by_action_size,
        include_x_origin=not by_action_size,
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
    plotted_instances = 0
    for instance in args.instances:
        output_stem = instance
        if args.m:
            output_stem += "_by_action_size"
        if args.full:
            output_stem += "_full"
        try:
            png_path, tex_path, pdf_path = plot_instance(
                instance,
                args.output_dir,
                output_stem=output_stem,
                by_action_size=args.m,
                n_bootstrap=args.bootstrap_samples,
                confidence_level=args.confidence_level,
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
