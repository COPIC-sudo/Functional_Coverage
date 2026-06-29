from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter


CONTROLLER_META: dict[str, dict[str, object]] = {
    "pi": {
        "label": "Weak TTC",
        "short": "Weak TTC",
        "color": "#8A8F98",
        "marker": "o",
        "linestyle": (0, (4, 2)),
    },
    "pi_tuned": {
        "label": "Strong TTC",
        "short": "Strong TTC",
        "color": "#0072B2",
        "marker": "s",
        "linestyle": "-",
    },
    "pi_sd": {
        "label": "Stopping-distance",
        "short": "Stopping-distance",
        "color": "#D55E00",
        "marker": "D",
        "linestyle": "-.",
    },
    "star": {
        "label": "Upper bound",
        "short": "Upper bound",
        "color": "#111111",
        "marker": "^",
        "linestyle": "-",
    },
    "pi_tuned_best": {
        "label": "Holdout-tuned TTC",
        "short": "Holdout-tuned TTC",
        "color": "#009E73",
        "marker": "P",
        "linestyle": "-",
    },
    "ablation_command_only": {
        "label": "Command-only",
        "short": "Command-only",
        "color": "#CC79A7",
        "marker": "X",
        "linestyle": (0, (1, 1)),
    },
    "ablation_trigger_only": {
        "label": "Trigger-only",
        "short": "Trigger-only",
        "color": "#E69F00",
        "marker": "v",
        "linestyle": (0, (3, 2, 1, 2)),
    },
    "ablation_relspeed_only": {
        "label": "Relative-speed-only",
        "short": "Relative-speed-only",
        "color": "#56B4E9",
        "marker": "<",
        "linestyle": (0, (5, 2)),
    },
}

MAIN_CONTROLLER_ORDER = ["pi", "pi_tuned", "pi_sd", "star"]
PRACTICAL_CONTROLLER_ORDER = ["pi", "pi_tuned", "pi_sd"]
PRACTICAL_HEATMAP_ORDER = ["pi_tuned", "pi_sd"]
ALL_FAMILY_HEATMAP_ORDER = ["pi", "pi_tuned", "pi_sd"]
CROSS_SITE_ORDER = [
    "ablation_command_only",
    "ablation_trigger_only",
    "ablation_relspeed_only",
    "pi",
    "pi_tuned_best",
    "pi_tuned",
    "pi_sd",
    "star",
]

DATASET_META = {
    "us101": {"label": "US-101", "marker": "o", "line": "-"},
    "i80": {"label": "I-80", "marker": "s", "line": "--"},
}

FAMILY_META = {
    # Reuse the muted gray / blue / orange publication palette used elsewhere
    # in the paper figures for a cleaner, more consistent print aesthetic.
    "braking_lead": {"label": "Braking lead", "color": "#0072B2"},
    "stationary_lead": {"label": "Stationary lead", "color": "#D55E00"},
    "slower_lead": {"label": "Slower lead", "color": "#8A8F98"},
}
FAMILY_ORDER = ["braking_lead", "stationary_lead", "slower_lead"]

PAPER_FIGURE_SPECS = {
    "fig01_main_coverage_frequency": "Main text · frequency-weighted coverage curves by site.",
    "fig02_main_coverage_severity": "Main text · severity-weighted coverage curves by site.",
    "fig03_main_coverage_practicality_frontier": "Main text · crash-mass coverage versus non-crash operational-burden costs.",
    "fig04_main_family_gap_heatmaps": "Main text · family-wise residual saturation gaps for practical controllers.",
    "figS01_supp_practicality_breakdown": "Supplement · detailed non-crash practicality metrics with conditional medians.",
    "figS02_supp_denominator_transparency": "Supplement · denominator transparency and population split.",
    "figS03_supp_family_composition": "Supplement · crash/non-crash family composition asymmetry.",
    "figS04_supp_bootstrap_saturation_intervals": "Supplement · bootstrap saturation intervals.",
    "figS05_supp_severity_profile_sensitivity": "Supplement · severity-profile sensitivity ranges.",
    "figS06_supp_cross_site_external_saturation": "Supplement · cross-site external saturation sanity check.",
    "figS07_supp_family_gap_heatmaps_with_base": "Supplement · family-wise residual saturation gaps including the simple baseline.",
}


@dataclass(frozen=True)
class FigureOutput:
    stem: str
    pdf_path: Path
    png_path: Path
    source_data_path: Path | None
    description: str


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.titlepad": 8,
            "figure.titlesize": 11,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "grid.color": "#B0B6BF",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.default": "regular",
        }
    )


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _resolve_paper_bundle_root(results_root: str | Path) -> Path:
    path = Path(results_root).expanduser().resolve()
    candidates = [
        path,
        path / "paper_bundle",
        path / "results" / "paper_bundle",
    ]
    for candidate in candidates:
        if (candidate / "assembled_sites" / "combined_main_table_frequency.csv").exists():
            return candidate
    tried = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate the paper bundle results root. Tried:\n"
        f"  - {tried}"
    )


def _ensure_output_dirs(output_root: Path) -> tuple[Path, Path, Path]:
    main_dir = output_root / "main"
    supp_dir = output_root / "supplement"
    source_dir = output_root / "source_data"
    for directory in (output_root, main_dir, supp_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return main_dir, supp_dir, source_dir


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def _format_dataset(dataset: str) -> str:
    return DATASET_META.get(dataset, {}).get("label", dataset)


def _format_family(family: str) -> str:
    return FAMILY_META.get(family, {}).get("label", family.replace("_", " ").title())


def _format_controller(controller: str) -> str:
    return str(CONTROLLER_META.get(controller, {}).get("label", controller))


def _human_count(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def _weighted_quantile(values: Iterable[float], weights: Iterable[float], quantile: float) -> float:
    values_arr = np.asarray(list(values), dtype=float)
    weights_arr = np.asarray(list(weights), dtype=float)
    mask = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr > 0)
    values_arr = values_arr[mask]
    weights_arr = weights_arr[mask]
    if values_arr.size == 0:
        return float("nan")
    order = np.argsort(values_arr)
    values_arr = values_arr[order]
    weights_arr = weights_arr[order]
    cumulative = np.cumsum(weights_arr)
    cutoff = quantile * weights_arr.sum()
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    idx = min(idx, values_arr.size - 1)
    return float(values_arr[idx])


def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    weights_arr = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights_arr) & (weights_arr > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.average(values[mask], weights=weights_arr[mask]))


def _source_and_save(fig: plt.Figure, stem: str, target_dir: Path, source_df: pd.DataFrame | None, source_dir: Path, png_dpi: int) -> FigureOutput:
    pdf_path = target_dir / f"{stem}.pdf"
    png_path = target_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=png_dpi)
    plt.close(fig)
    source_path: Path | None = None
    if source_df is not None:
        source_path = source_dir / f"{stem}.csv"
        source_df.to_csv(source_path, index=False)
    return FigureOutput(
        stem=stem,
        pdf_path=pdf_path,
        png_path=png_path,
        source_data_path=source_path,
        description=PAPER_FIGURE_SPECS.get(stem, ""),
    )




def _add_barh_value_label(
    ax: plt.Axes,
    value: float,
    y: float,
    x_min: float,
    x_max: float,
    text: str,
    inside_threshold: float = 0.92,
) -> None:
    """Place compact labels on horizontal bars without clipping."""
    if not math.isfinite(value):
        return
    span = x_max - x_min
    if span <= 0:
        return
    if value >= inside_threshold:
        x = max(x_min + 0.02 * span, value - 0.012 * span)
        ax.text(x, y, text, ha="right", va="center", fontsize=7.2, color="white")
    else:
        x = min(x_max - 0.002 * span, value + 0.012 * span)
        ax.text(x, y, text, ha="left", va="center", fontsize=7.2, color="#1F2530")


def _draw_interval_caps(ax: plt.Axes, y: float, low: float, high: float, color: str = "#1F2530") -> None:
    ax.hlines(y, low, high, color=color, linewidth=1.5, zorder=4)
    ax.vlines([low, high], y - 0.12, y + 0.12, color=color, linewidth=1.0, zorder=4)


def _default_tick(ax: plt.Axes, x: float, y: float, color: str = "#1F2530") -> None:
    ax.vlines(x, y - 0.28, y + 0.28, color=color, linewidth=1.2, zorder=5)


def _write_manifest(output_root: Path, outputs: list[FigureOutput]) -> None:
    rows = []
    for out in outputs:
        rows.append(
            {
                "figure_stem": out.stem,
                "description": out.description,
                "pdf_path": str(out.pdf_path.relative_to(output_root)),
                "png_path": str(out.png_path.relative_to(output_root)),
                "source_data_path": str(out.source_data_path.relative_to(output_root)) if out.source_data_path else "",
            }
        )
    pd.DataFrame(rows).to_csv(output_root / "figure_manifest.csv", index=False)


def _coverage_source_df(paper_bundle_root: Path, weight_mode: str) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    filename = f"coverage_{weight_mode}.csv"
    for dataset in ("us101", "i80"):
        path = paper_bundle_root / "sites" / dataset / filename
        df = _read_csv(path).copy()
        df.insert(0, "dataset", dataset)
        pieces.append(df)
    return pd.concat(pieces, ignore_index=True)


def _plot_coverage_curves(paper_bundle_root: Path, weight_mode: str, target_dir: Path, source_dir: Path, png_dpi: int, x_max: float = 4.0) -> FigureOutput:
    source_df = _coverage_source_df(paper_bundle_root, weight_mode)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True, constrained_layout=True)
    y_formatter = PercentFormatter(xmax=1.0, decimals=0)

    for ax, dataset in zip(axes, ("us101", "i80")):
        data = source_df[source_df["dataset"] == dataset].copy()
        data = data[data["delta_s"] <= x_max].copy()
        for controller in MAIN_CONTROLLER_ORDER:
            meta = CONTROLLER_META[controller]
            y = data[f"coverage_{controller}"]
            y_lo = data.get(f"coverage_{controller}_ci_low")
            y_hi = data.get(f"coverage_{controller}_ci_high")
            if y_lo is not None and y_hi is not None:
                ax.fill_between(
                    data["delta_s"],
                    y_lo,
                    y_hi,
                    color=str(meta["color"]),
                    alpha=0.10 if controller != "star" else 0.07,
                    linewidth=0.0,
                    zorder=1,
                )
            ax.plot(
                data["delta_s"],
                y,
                color=str(meta["color"]),
                linestyle=meta["linestyle"],
                label=str(meta["label"]),
                zorder=3 if controller == "star" else 2,
            )
            ax.plot(
                data["delta_s"].iloc[-1],
                y.iloc[-1],
                marker=str(meta["marker"]),
                color=str(meta["color"]),
                markersize=5.8,
                zorder=4,
            )
        ax.set_title(_format_dataset(dataset))
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Available lead time, $\\Delta$ (s)")
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_formatter(y_formatter)
        ax.grid(axis="y")
        _despine(ax)
    axes[0].set_ylabel("Crash-mass coverage")
    legend_handles = [
        Line2D([0], [0], color=str(CONTROLLER_META[c]["color"]), linestyle=CONTROLLER_META[c]["linestyle"], marker=str(CONTROLLER_META[c]["marker"]), label=str(CONTROLLER_META[c]["label"]))
        for c in MAIN_CONTROLLER_ORDER
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4)
    stem = "fig01_main_coverage_frequency" if weight_mode == "frequency" else "fig02_main_coverage_severity"
    return _source_and_save(fig, stem, target_dir, source_df, source_dir, png_dpi)


def _build_conditioned_practicality_summary(paper_bundle_root: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for dataset in ("us101", "i80"):
        for controller in PRACTICAL_CONTROLLER_ORDER:
            events_path = paper_bundle_root / "sites" / dataset / "non_crash_replay" / f"{controller}_noncrash_events.csv"
            if not events_path.exists():
                warnings.warn(f"Skipping missing non-crash replay file: {events_path}")
                continue
            events = _read_csv(events_path)
            weight_col = "weight_freq" if "weight_freq" in events.columns else None
            weights = events[weight_col] if weight_col else pd.Series(np.ones(len(events)), index=events.index)
            intervened = events["intervened"].astype(bool)
            comfort = events["comfort_exceed"].astype(bool)
            collision = events["collision_under_replay"].astype(bool)
            conditioned = events[intervened].copy()
            conditioned_weights = conditioned[weight_col] if weight_col else pd.Series(np.ones(len(conditioned)), index=conditioned.index)
            rows.append(
                {
                    "dataset": dataset,
                    "controller": controller,
                    "n_instances": int(len(events)),
                    "n_intervened": int(intervened.sum()),
                    "n_collisions": int(collision.sum()),
                    "n_conditioned": int(len(conditioned)),
                    "nuisance_intervention_rate": _weighted_mean(intervened.astype(float), weights),
                    "comfort_exceedance_rate": _weighted_mean(comfort.astype(float), weights),
                    "induced_collision_rate": _weighted_mean(collision.astype(float), weights),
                    "cond_median_peak_decel_mps2": _weighted_quantile(conditioned["peak_decel_mps2"], conditioned_weights, 0.50),
                    "cond_p90_peak_decel_mps2": _weighted_quantile(conditioned["peak_decel_mps2"], conditioned_weights, 0.90),
                    "cond_median_brake_duration_s": _weighted_quantile(conditioned["brake_duration_s"], conditioned_weights, 0.50),
                    "cond_p90_brake_duration_s": _weighted_quantile(conditioned["brake_duration_s"], conditioned_weights, 0.90),
                    "cond_median_min_residual_gap_m": _weighted_quantile(conditioned["min_residual_gap_m"], conditioned_weights, 0.50),
                }
            )
    if not rows:
        raise FileNotFoundError("No non-crash replay event files were found. Cannot build conditioned practicality summary.")
    return pd.DataFrame(rows)


def _plot_practicality_frontier(paper_bundle_root: Path, target_dir: Path, source_dir: Path, png_dpi: int) -> FigureOutput:
    conditioned = _build_conditioned_practicality_summary(paper_bundle_root)
    freq = _read_csv(paper_bundle_root / "assembled_sites" / "combined_main_table_frequency.csv")
    sev = _read_csv(paper_bundle_root / "assembled_sites" / "combined_main_table_severity.csv")

    freq = freq[freq["controller"].isin(PRACTICAL_CONTROLLER_ORDER + ["star"])].copy()
    sev = sev[sev["controller"].isin(PRACTICAL_CONTROLLER_ORDER + ["star"])].copy()
    freq_wide = freq.pivot(index="dataset", columns="controller", values="saturation")
    sev_wide = sev.pivot(index="dataset", columns="controller", values="saturation")

    plot_df = conditioned.merge(
        freq[freq["controller"].isin(PRACTICAL_CONTROLLER_ORDER)][["dataset", "controller", "saturation"]].rename(columns={"saturation": "frequency_saturation"}),
        on=["dataset", "controller"],
        how="left",
    ).merge(
        sev[sev["controller"].isin(PRACTICAL_CONTROLLER_ORDER)][["dataset", "controller", "saturation"]].rename(columns={"saturation": "severity_saturation"}),
        on=["dataset", "controller"],
        how="left",
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.25, 3.45), constrained_layout=True)
    panels = [
        {
            "ax": axes[0],
            "x": "nuisance_intervention_rate",
            "y": "frequency_saturation",
            "xlabel": "Non-crash nuisance-intervention rate",
            "ylabel": "Frequency-weighted crash coverage",
            "star_df": freq_wide,
            "title": "Coverage–nuisance",
            "xlim": (0.0, max(0.40, float(plot_df["nuisance_intervention_rate"].max()) * 1.10)),
        },
        {
            "ax": axes[1],
            "x": "comfort_exceedance_rate",
            "y": "severity_saturation",
            "xlabel": "Non-crash comfort-exceedance rate",
            "ylabel": "Severity-weighted crash coverage",
            "star_df": sev_wide,
            "title": "Coverage–comfort",
            "xlim": (0.0, max(0.12, float(plot_df["comfort_exceedance_rate"].max()) * 1.14)),
        },
    ]

    for panel in panels:
        ax = panel["ax"]
        for dataset in ("us101", "i80"):
            site = plot_df[plot_df["dataset"] == dataset].copy().sort_values(panel["x"])
            ax.plot(
                site[panel["x"]],
                site[panel["y"]],
                color="#B7BCC6",
                linewidth=1.35,
                linestyle=DATASET_META[dataset]["line"],
                zorder=1,
            )
            star_value = float(panel["star_df"].loc[dataset, "star"])
            ax.axhline(
                star_value,
                color="#D4D8DE" if dataset == "us101" else "#B7BCC6",
                linewidth=1.0,
                linestyle=DATASET_META[dataset]["line"],
                zorder=0,
            )

            for _, row in site.iterrows():
                meta = CONTROLLER_META[row["controller"]]
                ax.scatter(
                    row[panel["x"]],
                    row[panel["y"]],
                    s=44,
                    marker=DATASET_META[dataset]["marker"],
                    facecolor=str(meta["color"]),
                    edgecolor="white",
                    linewidth=0.9,
                    zorder=3,
                )

        ax.set_xlim(*panel["xlim"])
        ax.set_ylim(0.38, 1.01)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel(panel["ylabel"])
        ax.set_title(panel["title"])
        ax.grid(axis="both")
        _despine(ax)

    controller_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=str(CONTROLLER_META[c]["color"]), markeredgecolor="white", label=str(CONTROLLER_META[c]["label"]), markersize=6.4)
        for c in PRACTICAL_CONTROLLER_ORDER
    ]
    site_handles = [
        Line2D([0], [0], marker=DATASET_META[d]["marker"], color="#7A828D", label=_format_dataset(d), linestyle=DATASET_META[d]["line"], markersize=6)
        for d in ("us101", "i80")
    ]
    fig.legend(handles=controller_handles + site_handles, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=5)
    note = ""
    fig.text(0.01, -0.02, note, fontsize=7.3, color="#555C66")
    source_columns = [
        "dataset",
        "controller",
        "nuisance_intervention_rate",
        "comfort_exceedance_rate",
        "frequency_saturation",
        "severity_saturation",
        "cond_median_peak_decel_mps2",
        "cond_median_brake_duration_s",
        "induced_collision_rate",
    ]
    return _source_and_save(fig, "fig03_main_coverage_practicality_frontier", target_dir, plot_df[source_columns], source_dir, png_dpi)

def _plot_practicality_breakdown(paper_bundle_root: Path, target_dir: Path, source_dir: Path, png_dpi: int) -> FigureOutput:
    conditioned = _build_conditioned_practicality_summary(paper_bundle_root)
    conditioned = conditioned.copy()
    conditioned["dataset_label"] = conditioned["dataset"].map(_format_dataset)

    metrics = [
        ("nuisance_intervention_rate", "Nuisance-intervention rate", True),
        ("comfort_exceedance_rate", "Comfort-exceedance rate", True),
        ("cond_median_peak_decel_mps2", "Conditional median peak decel. (m/s²)", False),
        ("cond_median_brake_duration_s", "Conditional median brake duration (s)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 5.5), constrained_layout=True)
    axes = axes.flatten()
    datasets = ["us101", "i80"]
    x_centers = np.arange(len(datasets), dtype=float)
    width = 0.22
    offsets = np.array([-width, 0.0, width])

    for ax, (metric, title, is_percent) in zip(axes, metrics):
        for offset, controller in zip(offsets, PRACTICAL_CONTROLLER_ORDER):
            meta = CONTROLLER_META[controller]
            values = []
            for dataset in datasets:
                row = conditioned[(conditioned["dataset"] == dataset) & (conditioned["controller"] == controller)]
                values.append(float(row.iloc[0][metric]))
            bars = ax.bar(
                x_centers + offset,
                values,
                width=width * 0.92,
                color=str(meta["color"]),
                edgecolor="white",
                linewidth=0.8,
                label=str(meta["label"]),
                zorder=3,
            )
            for bar, value in zip(bars, values):
                if is_percent:
                    label = f"{value * 100:.1f}%"
                else:
                    label = f"{value:.2f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.008 if is_percent else 0.05),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )
        ax.set_xticks(x_centers)
        ax.set_xticklabels([_format_dataset(d) for d in datasets])
        ax.set_title(title)
        if is_percent:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis="y")
        _despine(ax)
    axes[0].set_ylim(0.0, 0.42)
    axes[1].set_ylim(0.0, 0.12)
    axes[2].set_ylim(0.0, max(3.2, float(conditioned["cond_median_peak_decel_mps2"].max()) * 1.25))
    axes[3].set_ylim(0.0, max(0.75, float(conditioned["cond_median_brake_duration_s"].max()) * 1.35))
    axes[2].set_xlabel("Crash-free replay subset")
    axes[3].set_xlabel("Crash-free replay subset")
    fig.legend(
        handles=[
            Line2D([0], [0], color=str(CONTROLLER_META[c]["color"]), lw=6, label=str(CONTROLLER_META[c]["label"]))
            for c in PRACTICAL_CONTROLLER_ORDER
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
    )
    note = ""
    fig.text(0.01, -0.02, note, fontsize=7.3, color="#555C66")
    return _source_and_save(fig, "figS01_supp_practicality_breakdown", target_dir, conditioned, source_dir, png_dpi)


def _heatmap_matrix(df: pd.DataFrame, controllers: list[str]) -> pd.DataFrame:
    plot_df = df[df["controller"].isin(controllers)].copy()
    plot_df["column_key"] = plot_df["dataset"].map(_format_dataset) + "\n" + plot_df["controller"].map(lambda c: str(CONTROLLER_META[c]["short"]))
    plot_df["family_label"] = plot_df["family"].map(_format_family)
    order_columns = []
    for dataset in ("us101", "i80"):
        for controller in controllers:
            order_columns.append(f"{_format_dataset(dataset)}\n{CONTROLLER_META[controller]['short']}")
    matrix = plot_df.pivot(index="family_label", columns="column_key", values="sat_gap_vs_star")
    family_index_order = [_format_family(f) for f in FAMILY_ORDER]
    matrix = matrix.reindex(index=family_index_order, columns=order_columns)
    return matrix * 100.0


def _draw_heatmap(ax: plt.Axes, matrix: pd.DataFrame, title: str, vmax: float, cbar: bool = False) -> None:
    cmap = LinearSegmentedColormap.from_list("paper_gap", ["#F6F8FB", "#AFC6E9", "#4477AA", "#133C73"])
    im = ax.imshow(matrix.values, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=0)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if not math.isfinite(value):
                text = "NA"
                color = "#20252B"
            else:
                text = f"{value:.1f}"
                color = "white" if value >= (0.60 * vmax) else "#1F2530"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.6, color=color)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Residual gap to Upper bound (pp)")


def _plot_family_gap_heatmaps(paper_bundle_root: Path, target_dir: Path, source_dir: Path, png_dpi: int, controllers: list[str], stem: str) -> FigureOutput:
    summary = _read_csv(paper_bundle_root / "assembled_sites" / "combined_family_summary.csv")
    summary = summary[summary["weight_mode"].isin(["frequency", "severity"])]
    freq_matrix = _heatmap_matrix(summary[summary["weight_mode"] == "frequency"], controllers)
    sev_matrix = _heatmap_matrix(summary[summary["weight_mode"] == "severity"], controllers)
    vmax = max(np.nanmax(freq_matrix.values), np.nanmax(sev_matrix.values))
    vmax = max(vmax, 10.0)

    fig, axes = plt.subplots(1, 2, figsize=(8.6 if len(controllers) == 2 else 10.2, 3.6), constrained_layout=True)
    _draw_heatmap(axes[0], freq_matrix, "Frequency-weighted residual saturation gap", vmax=vmax, cbar=False)
    _draw_heatmap(axes[1], sev_matrix, "Severity-weighted residual saturation gap", vmax=vmax, cbar=True)
    source_df = summary[summary["controller"].isin(controllers)][["dataset", "weight_mode", "family", "controller", "sat_gap_vs_star"]].copy()
    source_df["sat_gap_vs_star_pp"] = source_df["sat_gap_vs_star"] * 100.0
    return _source_and_save(fig, stem, target_dir, source_df, source_dir, png_dpi)

def _plot_family_gap_heatmaps_S07(
    paper_bundle_root: Path,
    target_dir: Path,
    source_dir: Path,
    png_dpi: int,
    controllers: list[str],
    stem: str,
) -> FigureOutput:
    summary = _read_csv(paper_bundle_root / "assembled_sites" / "combined_family_summary.csv")
    summary = summary[summary["weight_mode"].isin(["frequency", "severity"])]

    freq_matrix = _heatmap_matrix(summary[summary["weight_mode"] == "frequency"], controllers)
    sev_matrix = _heatmap_matrix(summary[summary["weight_mode"] == "severity"], controllers)

    vmax = max(np.nanmax(freq_matrix.values), np.nanmax(sev_matrix.values))
    vmax = max(vmax, 10.0)

    # Widen the figure to give bottom labels more room.
    fig_w = 9.6 if len(controllers) == 2 else 12.8
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 4.6), constrained_layout=False)

    _draw_heatmap(axes[0], freq_matrix, "Frequency-weighted residual saturation gap", vmax=vmax, cbar=False)
    _draw_heatmap(axes[1], sev_matrix, "Severity-weighted residual saturation gap", vmax=vmax, cbar=True)

    # Reformat x-axis labels to avoid overlap.
    def _tidy_xticklabels(ax):
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        labels = [
            s.replace("Stopping-distance", "Stopping-\ndistance")
             .replace("Holdout-tuned TTC", "Holdout-tuned\nTTC")
             .replace("Relative-speed-only", "Relative-speed-\nonly")
             .replace("US-101 ", "US-101\n")
             .replace("I-80 ", "I-80\n")
            for s in labels
        ]
        ax.set_xticklabels(
            labels,
            fontsize=9,
            rotation=15,
            ha="right",
            rotation_mode="anchor",
        )
        ax.tick_params(axis="x", pad=6)

    _tidy_xticklabels(axes[0])
    _tidy_xticklabels(axes[1])

    # Reserve space for bottom labels and subplot spacing.
    fig.subplots_adjust(bottom=0.24, wspace=0.18)

    source_df = summary[summary["controller"].isin(controllers)][
        ["dataset", "weight_mode", "family", "controller", "sat_gap_vs_star"]
    ].copy()
    source_df["sat_gap_vs_star_pp"] = source_df["sat_gap_vs_star"] * 100.0

    return _source_and_save(fig, stem, target_dir, source_df, source_dir, png_dpi)





def _denominator_share_reference_text(stage_key: str) -> str | None:
    """Return the explicit reference group used by the denominator summary percentages.

    The transparency CSV reports ``share_of_previous`` relative to the full pipeline,
    including hidden intermediate stages that are not drawn in Fig. S02. To keep the
    four-bar extraction funnel while preserving the original summary semantics, the
    figure labels state the true comparison object explicitly rather than using the
    ambiguous phrase "of previous".
    """
    reference_labels = {
        "paired_rows": "merged rows",
        "candidate_rows": "same-lane rows",
        "base_episodes_retained": "pre-duration episodes",
    }
    return reference_labels.get(stage_key)

def _plot_denominator_transparency(paper_bundle_root: Path, target_dir: Path, source_dir: Path, png_dpi: int) -> FigureOutput:
    df = _read_csv(paper_bundle_root / "assembled_sites" / "combined_denominator_transparency_summary.csv")
    extraction_order = [
        "raw_standardized_rows",
        "paired_rows",
        "candidate_rows",
        "base_episodes_retained",
    ]
    population_order = ["replicated_instances", "crash_instances", "noncrash_instances"]
    extraction_labels = {
        "raw_standardized_rows": "Raw standardized rows",
        "paired_rows": "Rows with valid lead attachment",
        "candidate_rows": "Candidate rear-end rows",
        "base_episodes_retained": "Retained base episodes",
    }
    population_labels = {
        "replicated_instances": "Replicated instances",
        "crash_instances": "$\\mathcal{S}_{cr}$ crash instances",
        "noncrash_instances": "$\\mathcal{S}_{nc}$ non-crash instances",
    }

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.6), constrained_layout=True)
    for row_idx, dataset in enumerate(("us101", "i80")):
        site = df[df["dataset"] == dataset].copy()
        left = axes[row_idx, 0]
        right = axes[row_idx, 1]

        extraction = site.set_index("stage_key").loc[extraction_order].reset_index()
        y = np.arange(len(extraction))[::-1]
        colors = ["#9DB7D8", "#7FA6D1", "#5E90C7", "#4477AA"]
        left.barh(y, extraction["count"], color=colors, edgecolor="white", linewidth=0.8, zorder=3)
        left.set_yticks(y)
        left.set_yticklabels([extraction_labels[s] for s in extraction["stage_key"]])
        left.set_xscale("log")
        left.set_xlabel("Count (log scale)")
        left.set_title(f"{_format_dataset(dataset)} · extraction funnel")
        left.grid(axis="x")
        _despine(left)
        for yi, (_, stage) in zip(y, extraction.iterrows()):
            x = float(stage["count"])
            retention = stage["share_of_previous"]
            stage_key = str(stage["stage_key"])
            label = _human_count(x)
            if math.isfinite(float(x)):
                left.text(x * 1.05, yi + 0.11, label, va="center", fontsize=7.5)
            if pd.notna(retention):
                reference_text = _denominator_share_reference_text(stage_key)
                if reference_text:
                    retention_text = f"{retention * 100:.1f}% of {reference_text}"
                else:
                    retention_text = f"{retention * 100:.1f}% of previous"
                left.text(x * 1.05, yi - 0.13, retention_text, va="center", fontsize=7.2, color="#56606D")

        population = site.set_index("stage_key").loc[population_order].reset_index()
        y2 = np.arange(len(population))[::-1]
        pop_colors = ["#A0A7B3", "#D55E00", "#228833"]
        right.barh(y2, population["count"], color=pop_colors, edgecolor="white", linewidth=0.8, zorder=3)
        right.set_yticks(y2)
        right.set_yticklabels([population_labels[s] for s in population["stage_key"]])
        right.set_xscale("log")
        right.set_xlabel("Count (log scale)")
        right.set_title(f"{_format_dataset(dataset)} · modeled population split")
        right.grid(axis="x")
        _despine(right)
        for yi, (_, stage) in zip(y2, population.iterrows()):
            x = float(stage["count"])
            base_n = stage.get("n_base_scenarios")
            label = _human_count(x)
            right.text(x * 1.05, yi + 0.11, label, va="center", fontsize=7.5)
            if pd.notna(base_n):
                right.text(x * 1.05, yi - 0.13, f"base episodes: {_human_count(base_n)}", va="center", fontsize=7.2, color="#56606D")

    note = ""
    fig.text(0.01, -0.02, note, fontsize=7.3, color="#555C66")
    return _source_and_save(fig, "figS02_supp_denominator_transparency", target_dir, df, source_dir, png_dpi)


def _plot_family_composition(paper_bundle_root: Path, target_dir: Path, source_dir: Path, png_dpi: int) -> FigureOutput:
    df = _read_csv(paper_bundle_root / "assembled_sites" / "combined_family_composition.csv")
    rows = []
    for dataset in ("us101", "i80"):
        crash = df[(df["dataset"] == dataset) & (df["subset"] == "crash")]
        noncrash = df[(df["dataset"] == dataset) & (df["subset"] == "noncrash")]
        for subset_name, subset_df, share_col in [
            ("Crash subset (severity share)", crash, "severity_share"),
            ("Non-crash subset (frequency share)", noncrash, "freq_share"),
        ]:
            for family in FAMILY_ORDER:
                match = subset_df.loc[subset_df["family"] == family, share_col]
                value = float(match.iloc[0]) if not match.empty else 0.0
                rows.append(
                    {
                        "dataset": dataset,
                        "subset_display": subset_name,
                        "family": family,
                        "share": value,
                    }
                )
    plot_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, ("us101", "i80")):
        site = plot_df[plot_df["dataset"] == dataset]
        bottoms = np.zeros(2, dtype=float)
        x = np.arange(2)
        for family in FAMILY_ORDER:
            family_df = site[site["family"] == family].sort_values("subset_display")
            values = family_df["share"].to_numpy()
            bars = ax.bar(
                x,
                values,
                bottom=bottoms,
                color=FAMILY_META[family]["color"],
                edgecolor="white",
                linewidth=0.8,
                label=FAMILY_META[family]["label"],
            )
            for bar, value, bottom in zip(bars, values, bottoms):
                if value >= 0.08:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom + value / 2,
                        f"{value * 100:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                    )
            bottoms += values
        ax.set_xticks(x)
        ax.set_xticklabels(["Crash\n(severity)", "Non-crash\n(frequency)"])
        ax.set_title(_format_dataset(dataset))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis="y")
        _despine(ax)
    axes[0].set_ylabel("Subset share")
    family_handles = [
        Line2D([0], [0], color=FAMILY_META[f]["color"], lw=6, label=FAMILY_META[f]["label"])
        for f in FAMILY_ORDER
    ]
    fig.legend(handles=family_handles, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=3)
    note = ""
    fig.text(0.01, -0.02, note, fontsize=7.3, color="#555C66")
    return _source_and_save(fig, "figS03_supp_family_composition", target_dir, plot_df, source_dir, png_dpi)

def _plot_bootstrap_saturation_intervals(paper_bundle_root: Path, target_dir: Path, source_dir: Path, png_dpi: int) -> FigureOutput:
    df = _read_csv(paper_bundle_root / "assembled_sites" / "combined_bootstrap_interval_summary.csv")
    df = df[df["metric"] == "saturation"].copy()
    controller_order = ["star", "pi_sd", "pi_tuned", "pi"]
    y_positions = np.arange(len(controller_order))[::-1]
    x_min, x_max = 0.35, 1.01

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.2), sharex=True, constrained_layout=True)
    for row_idx, dataset in enumerate(("us101", "i80")):
        for col_idx, weight_mode in enumerate(("frequency", "severity")):
            ax = axes[row_idx, col_idx]
            subset = df[(df["dataset"] == dataset) & (df["weight_mode"] == weight_mode)].copy()
            subset["controller"] = pd.Categorical(subset["controller"], categories=controller_order, ordered=True)
            subset = subset.sort_values("controller", ascending=False)

            for yi, controller in zip(y_positions, controller_order):
                row = subset[subset["controller"] == controller]
                if row.empty:
                    continue
                row = row.iloc[0]
                color = str(CONTROLLER_META[controller]["color"])
                value = float(row["value"])
                ax.barh(
                    yi,
                    value,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    height=0.62,
                    zorder=3,
                )
                _draw_interval_caps(ax, yi, float(row["ci_low"]), float(row["ci_high"]))
                _add_barh_value_label(ax, value, yi, x_min, x_max, f"{value * 100:.1f}%")

            ax.set_title(f"{_format_dataset(dataset)} · {'frequency' if weight_mode == 'frequency' else 'severity'} weighting")
            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax.grid(axis="x")
            _despine(ax)

            ax.set_yticks(y_positions)
            if col_idx == 0:
                ax.set_yticklabels([str(CONTROLLER_META[c]["short"]) for c in controller_order])
                ax.set_ylabel("Controller")
            else:
                ax.set_yticklabels([])

    note = ""
    fig.text(0.01, -0.02, note, fontsize=7.3, color="#555C66")
    return _source_and_save(fig, "figS04_supp_bootstrap_saturation_intervals", target_dir, df, source_dir, png_dpi)

def _plot_severity_profile_sensitivity(paper_bundle_root: Path, target_dir: Path, source_dir: Path, png_dpi: int) -> FigureOutput:
    df = _read_csv(paper_bundle_root / "assembled_sites" / "combined_severity_profile_range_summary.csv")
    df = df[df["metric"].isin(["saturation", "sat_gap_vs_star"])].copy()
    controller_order = ["star", "pi_sd", "pi_tuned", "pi"]
    y_positions = np.arange(len(controller_order))[::-1]

    sat_min = max(0.35, float(df.loc[df["metric"] == "saturation", "min_value"].min()) - 0.02)
    sat_max = min(1.01, float(df.loc[df["metric"] == "saturation", "max_value"].max()) + 0.02)
    gap_min = min(0.0, float(df.loc[df["metric"] == "sat_gap_vs_star", "min_value"].min()))
    gap_max = float(df.loc[df["metric"] == "sat_gap_vs_star", "max_value"].max()) * 1.10
    gap_max = max(gap_max, 0.12)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.6), sharex="col", constrained_layout=True)
    metric_specs = [
        ("saturation", "Saturation", PercentFormatter(xmax=1.0, decimals=0), sat_min, sat_max),
        ("sat_gap_vs_star", "Residual gap to Upper bound", FuncFormatter(lambda x, _pos: f"{x * 100:.0f} pp"), gap_min, gap_max),
    ]

    for row_idx, dataset in enumerate(("us101", "i80")):
        for col_idx, (metric, title, formatter, xmin, xmax) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            subset = df[(df["dataset"] == dataset) & (df["metric"] == metric)].copy()
            subset["controller"] = pd.Categorical(subset["controller"], categories=controller_order, ordered=True)
            subset = subset.sort_values("controller", ascending=False)

            for yi, controller in zip(y_positions, controller_order):
                row = subset[subset["controller"] == controller]
                if row.empty:
                    continue
                row = row.iloc[0]
                left = float(row["min_value"])
                right = float(row["max_value"])
                default = float(row["default_value"])
                width = max(0.0, right - left)
                if width > 0:
                    ax.barh(
                        yi,
                        width,
                        left=left,
                        color=str(CONTROLLER_META[controller]["color"]),
                        edgecolor="white",
                        linewidth=0.8,
                        height=0.62,
                        zorder=3,
                    )
                _default_tick(ax, default, yi)

            ax.set_title(f"{_format_dataset(dataset)} · {title}")
            ax.set_xlim(xmin, xmax)
            ax.xaxis.set_major_formatter(formatter)
            ax.grid(axis="x")
            _despine(ax)

            ax.set_yticks(y_positions)
            if col_idx == 0:
                ax.set_yticklabels([str(CONTROLLER_META[c]["short"]) for c in controller_order])
                ax.set_ylabel("Controller")
            else:
                ax.set_yticklabels([])

    legend_handles = [
        Line2D([0], [0], color="#1F2530", lw=1.2, label="Default profile tick"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=1)
    # note = "Colored bars span the minimum-to-maximum value over the tested severity profiles; black ticks indicate the default composite profile."
    note = ""
    fig.text(0.01, -0.02, note, fontsize=7.3, color="#555C66")
    return _source_and_save(fig, "figS05_supp_severity_profile_sensitivity", target_dir, df, source_dir, png_dpi)

def _plot_cross_site_external_saturation(paper_bundle_root: Path, target_dir: Path, source_dir: Path, png_dpi: int) -> FigureOutput:
    df = _read_csv(paper_bundle_root / "cross_site" / "assembled" / "cross_site_holdout_summary.csv")
    df = df[(df["split"] == "external_test") & (df["metric"] == "saturation")].copy()
    controller_order = [
        "star",
        "pi_sd",
        "pi_tuned",
        "pi_tuned_best",
        "pi",
        "ablation_relspeed_only",
        "ablation_trigger_only",
        "ablation_command_only",
    ]
    y_positions = np.arange(len(controller_order))[::-1]
    x_min, x_max = 0.35, 1.01

    fig, axes = plt.subplots(2, 2, figsize=(9.1, 7.0), sharex=True, constrained_layout=True)
    for row_idx, direction in enumerate(("us101_to_i80", "i80_to_us101")):
        for col_idx, weight_mode in enumerate(("frequency", "severity")):
            ax = axes[row_idx, col_idx]
            subset = df[(df["direction"] == direction) & (df["weight_mode"] == weight_mode)].copy()
            subset["controller"] = pd.Categorical(subset["controller"], categories=controller_order, ordered=True)
            subset = subset.sort_values("controller", ascending=False)

            for yi, controller in zip(y_positions, controller_order):
                row = subset[subset["controller"] == controller]
                if row.empty:
                    continue
                row = row.iloc[0]
                value = float(row["value"])
                ax.barh(
                    yi,
                    value,
                    color=str(CONTROLLER_META[controller]["color"]),
                    edgecolor="white",
                    linewidth=0.8,
                    height=0.62,
                    zorder=3,
                )
                _add_barh_value_label(ax, value, yi, x_min, x_max, f"{value * 100:.1f}%")

            direction_label = "US-101 → I-80" if direction == "us101_to_i80" else "I-80 → US-101"
            ax.set_title(f"{direction_label} · {'frequency' if weight_mode == 'frequency' else 'severity'} weighting")
            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax.grid(axis="x")
            _despine(ax)

            ax.set_yticks(y_positions)
            if col_idx == 0:
                ax.set_yticklabels([str(CONTROLLER_META[c]["short"]) for c in controller_order])
                ax.set_ylabel("External-test controller")
            else:
                ax.set_yticklabels([])

    note = ""
    fig.text(0.01, -0.02, note, fontsize=7.3, color="#555C66")
    return _source_and_save(fig, "figS06_supp_cross_site_external_saturation", target_dir, df, source_dir, png_dpi)

def generate_all_paper_figures(results_root: str | Path, output_root: str | Path | None = None, png_dpi: int = 600, x_max: float = 4.0) -> list[FigureOutput]:
    _apply_style()
    paper_bundle_root = _resolve_paper_bundle_root(results_root)
    output_root = Path(output_root).expanduser().resolve() if output_root else paper_bundle_root / "paper_figures"
    main_dir, supp_dir, source_dir = _ensure_output_dirs(output_root)

    outputs: list[FigureOutput] = []
    outputs.append(_plot_coverage_curves(paper_bundle_root, "frequency", main_dir, source_dir, png_dpi, x_max=x_max))
    outputs.append(_plot_coverage_curves(paper_bundle_root, "severity", main_dir, source_dir, png_dpi, x_max=x_max))
    outputs.append(_plot_practicality_frontier(paper_bundle_root, main_dir, source_dir, png_dpi))
    outputs.append(_plot_family_gap_heatmaps(paper_bundle_root, main_dir, source_dir, png_dpi, PRACTICAL_HEATMAP_ORDER, "fig04_main_family_gap_heatmaps"))

    outputs.append(_plot_practicality_breakdown(paper_bundle_root, supp_dir, source_dir, png_dpi))
    outputs.append(_plot_denominator_transparency(paper_bundle_root, supp_dir, source_dir, png_dpi))
    outputs.append(_plot_family_composition(paper_bundle_root, supp_dir, source_dir, png_dpi))
    outputs.append(_plot_bootstrap_saturation_intervals(paper_bundle_root, supp_dir, source_dir, png_dpi))
    outputs.append(_plot_severity_profile_sensitivity(paper_bundle_root, supp_dir, source_dir, png_dpi))

    cross_site_root = paper_bundle_root / "cross_site" / "assembled" / "cross_site_holdout_summary.csv"
    if cross_site_root.exists():
        outputs.append(_plot_cross_site_external_saturation(paper_bundle_root, supp_dir, source_dir, png_dpi))
    else:
        warnings.warn("Cross-site assembled results not found. Skipping cross-site external saturation figure.")

    outputs.append(_plot_family_gap_heatmaps_S07(paper_bundle_root, supp_dir, source_dir, png_dpi, ALL_FAMILY_HEATMAP_ORDER, "figS07_supp_family_gap_heatmaps_with_base"))

    _write_manifest(output_root, outputs)
    return outputs
