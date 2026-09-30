"""Utilities for plotting regret in the example scripts."""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_regret_plot(
    rewards: np.ndarray,
    actions_by_label: dict[str, np.ndarray],
    action_size: int,
    output_path: Path,
    title: str,
) -> tuple[int, ...]:
    """Plot cumulative regret curves against the best fixed action."""
    best_subset = best_fixed_subset(rewards, action_size)
    benchmark_rewards = rewards[:, best_subset].sum(axis=1)

    figure, axes = plt.subplots()
    timesteps = np.arange(1, rewards.shape[0] + 1)
    for label, actions in actions_by_label.items():
        selected_rewards = np.sum(rewards * actions, axis=1)
        cumulative_regret = np.cumsum(
            benchmark_rewards - selected_rewards
        )
        axes.plot(timesteps, cumulative_regret, label=label)
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.set_xlabel("Timestep")
    axes.set_ylabel("Cumulative regret")
    axes.set_title(title)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return best_subset


def best_fixed_subset(
    rewards: np.ndarray, action_size: int
) -> tuple[int, ...]:
    """Return the fixed subset with the largest reward over the horizon."""
    subsets = combinations(range(rewards.shape[1]), action_size)
    return max(subsets, key=lambda subset: rewards[:, subset].sum())


def cumulative_regret(
    rewards: np.ndarray,
    actions: np.ndarray,
    best_subset: tuple[int, ...],
) -> np.ndarray:
    """Compute cumulative regret against one fixed hindsight action."""
    if actions.shape != rewards.shape:
        raise ValueError("actions and rewards must have the same shape")
    benchmark_rewards = rewards[:, best_subset].sum(axis=1)
    selected_rewards = np.sum(rewards * actions, axis=1)
    return np.cumsum(benchmark_rewards - selected_rewards)


def blockwise_cumulative_regret(
    rewards: np.ndarray,
    actions: np.ndarray,
    action_size: int,
    block_sizes: np.ndarray,
) -> np.ndarray:
    """Compare with the best fixed subset in each hindsight block."""
    if actions.shape != rewards.shape:
        raise ValueError("actions and rewards must have the same shape")
    sizes = np.asarray(block_sizes)
    if (sizes.ndim != 1
            or np.any(sizes <= 0)
            or np.any(sizes != np.floor(sizes))
            or sizes.sum() != rewards.shape[0]):
        raise ValueError("block_sizes must be positive and sum to horizon")

    benchmark_rewards = np.empty(rewards.shape[0])
    start = 0
    for size in sizes.astype(int):
        stop = start + size
        block = rewards[start:stop]
        best_subset = best_fixed_subset(block, action_size)
        benchmark_rewards[start:stop] = block[:, best_subset].sum(axis=1)
        start = stop

    selected_rewards = np.sum(rewards * actions, axis=1)
    return np.cumsum(benchmark_rewards - selected_rewards)


def seed_column_names(seeds: tuple[int, ...]) -> list[str]:
    """Make unique CSV columns, including when a seed is repeated."""
    totals = {seed: seeds.count(seed) for seed in set(seeds)}
    seen: dict[int, int] = {}
    names = []
    for seed in seeds:
        seen[seed] = seen.get(seed, 0) + 1
        suffix = f"_run_{seen[seed]}" if totals[seed] > 1 else ""
        names.append(f"seed_{seed}{suffix}")
    return names


def save_regret_csv(
    regrets: np.ndarray,
    seeds: tuple[int, ...],
    output_path: Path,
) -> None:
    """Save a timestep-by-run cumulative-regret matrix."""
    if regrets.ndim != 2 or regrets.shape[1] != len(seeds):
        raise ValueError("regrets must have one column per seed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack((np.arange(1, regrets.shape[0] + 1), regrets))
    header = ",".join(("timestep", *seed_column_names(seeds)))
    np.savetxt(
        output_path,
        rows,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", *(["%.18g"] * regrets.shape[1])],
    )


def bootstrap_weights(
    n_runs: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw bootstrap resamples and return their normalized run counts."""
    if n_runs < 1 or n_bootstrap < 1:
        raise ValueError("n_runs and n_bootstrap must be positive")
    draws = rng.integers(n_runs, size=(n_bootstrap, n_runs))
    weights = np.zeros((n_bootstrap, n_runs), dtype=float)
    for run in range(n_runs):
        weights[:, run] = np.count_nonzero(draws == run, axis=1) / n_runs
    return weights


def bootstrap_confidence_region(
    regrets: np.ndarray,
    weights: np.ndarray,
    confidence_level: float,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a pointwise percentile bootstrap confidence region."""
    if regrets.ndim != 2 or weights.ndim != 2:
        raise ValueError("regrets and weights must be matrices")
    if regrets.shape[1] != weights.shape[1]:
        raise ValueError("bootstrap weights must match the number of runs")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    tail = (1.0 - confidence_level) / 2.0
    lower = np.empty(regrets.shape[0])
    upper = np.empty(regrets.shape[0])
    for start in range(0, regrets.shape[0], chunk_size):
        stop = min(start + chunk_size, regrets.shape[0])
        bootstrap_means = regrets[start:stop] @ weights.T
        lower[start:stop], upper[start:stop] = np.quantile(
            bootstrap_means, (tail, 1.0 - tail), axis=1
        )
    return lower, upper


def save_bootstrap_regret_plot(
    regrets_by_label: dict[str, np.ndarray],
    output_path: Path,
    title: str | None = None,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 0,
    y_label: str = "Cumulative regret",
    x_values: np.ndarray | None = None,
    x_label: str = "Timestep",
    show_all_x_ticks: bool = False,
    include_x_origin: bool = True,
) -> None:
    """Plot mean cumulative regret with pointwise bootstrap regions."""
    if not regrets_by_label:
        raise ValueError("at least one algorithm is required")
    shapes = {regrets.shape for regrets in regrets_by_label.values()}
    if len(shapes) != 1:
        raise ValueError("all regret matrices must have the same shape")
    n_points, n_runs = next(iter(shapes))
    plot_x = _validate_x_values(x_values, n_points)
    weights = bootstrap_weights(
        n_runs, n_bootstrap, np.random.default_rng(bootstrap_seed)
    )

    figure, axes = plt.subplots()
    markers = ("o", "s", "^", "D")
    marker_stride = max(1, int(np.ceil(n_points / 12)))
    y_min = np.inf
    y_max = -np.inf
    for plot_index, (label, regrets) in enumerate(regrets_by_label.items()):
        mean = regrets.mean(axis=1)
        lower, upper = bootstrap_confidence_region(
            regrets, weights, confidence_level
        )
        y_min = min(y_min, float(lower.min()))
        y_max = max(y_max, float(upper.max()))
        line, = axes.plot(
            plot_x,
            mean,
            label=label,
            marker=markers[plot_index % len(markers)],
            markevery=marker_stride,
            markersize=7.5,
            markerfacecolor="white",
            markeredgewidth=1,
        )
        axes.fill_between(
            plot_x,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.2,
            linewidth=0,
        )
    y_bottom, y_top = _limits_including_origin(y_min, y_max)
    axes.set_xlim(*_x_limits(plot_x, include_x_origin))
    axes.set_ylim(y_bottom, y_top)
    if show_all_x_ticks:
        axes.set_xticks(plot_x)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def save_tikz_regret_plot(
    regrets_by_label: dict[str, np.ndarray],
    output_path: Path,
    title: str | None = None,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 0,
    y_label: str = "Cumulative regret",
    max_points: int = 1200,
    x_values: np.ndarray | None = None,
    x_label: str = "Timestep",
    show_all_x_ticks: bool = False,
    include_x_origin: bool = True,
) -> None:
    """Write a standalone PGFPlots version of a bootstrap regret plot."""
    if not regrets_by_label:
        raise ValueError("at least one algorithm is required")
    shapes = {regrets.shape for regrets in regrets_by_label.values()}
    if len(shapes) != 1:
        raise ValueError("all regret matrices must have the same shape")
    n_points, n_runs = next(iter(shapes))
    all_x_values = _validate_x_values(x_values, n_points)
    weights = bootstrap_weights(
        n_runs, n_bootstrap, np.random.default_rng(bootstrap_seed)
    )
    indices = np.unique(
        np.linspace(0, n_points - 1, min(n_points, max_points), dtype=int)
    )
    plot_x = all_x_values[indices]
    colors = (
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
    )
    markers = ("*", "square*", "triangle*", "diamond*")
    marker_stride = max(1, int(np.ceil(len(indices) / 12)))
    series = []
    y_min = np.inf
    y_max = -np.inf
    for label, regrets in regrets_by_label.items():
        mean = regrets.mean(axis=1)
        lower, upper = bootstrap_confidence_region(
            regrets, weights, confidence_level
        )
        y_min = min(y_min, float(lower.min()))
        y_max = max(y_max, float(upper.max()))
        series.append((label, mean, lower, upper))
    y_bottom, y_top = _limits_including_origin(y_min, y_max)
    x_left, x_right = _x_limits(plot_x, include_x_origin)

    lines = [
        r"\documentclass{article}",
        r"\usepackage[paperwidth=7in,paperheight=5in,margin=0.15in]{geometry}",
        r"\usepackage{pgfplots}",
        r"\usepgfplotslibrary{fillbetween}",
        r"\pgfplotsset{compat=1.18}",
    ]
    for index, color in enumerate(colors):
        lines.append(
            rf"\definecolor{{plotcolor{index}}}{{RGB}}{{{color[0]},{color[1]},{color[2]}}}"
        )
    lines.extend(
        [
            r"\pagestyle{empty}",
            r"\begin{document}",
            r"\noindent\begin{tikzpicture}",
            r"\begin{axis}[",
            r"width=6.2in,height=4.15in,",
            rf"xlabel={{{_tex_escape(x_label)}}},",
            rf"ylabel={{{_tex_escape(y_label)}}},",
            rf"xmin={x_left:.10g},",
            rf"xmax={x_right:.10g},",
            rf"ymin={y_bottom:.10g},",
            rf"ymax={y_top:.10g},",
            r"enlarge x limits=false,",
            r"enlarge y limits=false,",
            r"legend pos=north west,",
            r"legend cell align={left},",
            r"axis lines=box,",
            r"tick align=outside,",
        ]
    )
    if show_all_x_ticks:
        ticks = ",".join(f"{float(value):.10g}" for value in plot_x)
        lines.append(rf"xtick={{{ticks}}},")
    lines.append(r"]")
    for plot_index, (label, mean, lower, upper) in enumerate(series):
        color = f"plotcolor{plot_index % len(colors)}"
        marker = markers[plot_index % len(markers)]
        upper_path = f"upper{plot_index}"
        lower_path = f"lower{plot_index}"
        lines.extend(
            [
                rf"\addplot[name path={upper_path},draw=none,forget plot] coordinates {{{_tikz_coordinates(plot_x, upper[indices])}}};",
                rf"\addplot[name path={lower_path},draw=none,forget plot] coordinates {{{_tikz_coordinates(plot_x, lower[indices])}}};",
                rf"\addplot[fill={color},fill opacity=0.2,draw=none,forget plot] fill between[of={upper_path} and {lower_path}];",
                rf"\addplot[color={color},thick,mark={marker},mark repeat={marker_stride},mark options={{fill=white,solid}},mark size=3pt] coordinates {{{_tikz_coordinates(plot_x, mean[indices])}}};",
                rf"\addlegendentry{{{_tex_escape(label)}}}",
            ]
        )
    lines.extend(
        [
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\end{document}",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _limits_including_origin(
    data_min: float, data_max: float
) -> tuple[float, float]:
    """Return tight plot limits that include zero without clipping data."""
    lower = min(0.0, data_min)
    upper = max(0.0, data_max)
    span = upper - lower
    if span == 0.0:
        return 0.0, 1.0
    padding = 0.02 * span
    bottom = lower - padding if lower < 0.0 else 0.0
    top = upper + padding if upper > 0.0 else 0.0
    return bottom, top


def _validate_x_values(
    x_values: np.ndarray | None, n_points: int
) -> np.ndarray:
    if x_values is None:
        return np.arange(1, n_points + 1)
    values = np.asarray(x_values)
    if (
        values.ndim != 1
        or len(values) != n_points
        or not np.all(np.isfinite(values))
        or np.any(np.diff(values) <= 0)
    ):
        raise ValueError(
            "x_values must be finite, strictly increasing, and match the data"
        )
    return values


def _x_limits(
    x_values: np.ndarray, include_origin: bool
) -> tuple[float, float]:
    first = float(x_values[0])
    last = float(x_values[-1])
    if include_origin:
        return min(0.0, first), max(0.0, last)
    span = last - first
    padding = 0.02 * span if span else max(0.02 * abs(first), 0.5)
    return first - padding, last + padding


def _tikz_coordinates(x_values: np.ndarray, y_values: np.ndarray) -> str:
    return " ".join(
        f"({int(x_value)},{float(y_value):.10g})"
        for x_value, y_value in zip(x_values, y_values)
    )


def _tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)
