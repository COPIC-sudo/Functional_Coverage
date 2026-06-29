#!/usr/bin/env python3
"""Generate publication-ready figures for the RSS/highD robustness experiment.

This script reads either the zipped result bundle (e.g.
``rss_highd_zero_tune_results_v1.0.zip``) or the unzipped results root
(``rss_highd_zero_tune/``) and produces three figure files in both PDF and PNG
formats:

1. Main-text overlay figure:
   ``paper_figures/main/fig05_main_rss_highd_transfer.(pdf|png)``
2. Supplementary transfer curves:
   ``paper_figures/supplement/figS08_supp_rss_highd_curves.(pdf|png)``
3. Supplementary highD stratum summary:
   ``paper_figures/supplement/figS09_supp_highd_strata.(pdf|png)``

The script also exports tidy source-data CSV files to
``paper_figures/source_data``.

Usage examples
--------------
Run from the LaTeX paper root:

    python scripts/make_rss_highd_transfer_figures_v1.0.0.py \
        --results /path/to/rss_highd_zero_tune_results_v1.0.zip

Or point to an unzipped results directory:

    python scripts/make_rss_highd_transfer_figures_v1.0.0.py \
        --results /path/to/rss_highd_zero_tune \
        --paper-root .
"""

from __future__ import annotations

import argparse
import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


# ---------------------------------------------------------------------------
# Global plotting style
# ---------------------------------------------------------------------------
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.18,
        "grid.color": "0.25",
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavuserif",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


CONTROLLER_ORDER = ["star", "pi", "pi_tuned", "pi_sd", "rss_longitudinal"]
STRATUM_CONTROLLER_ORDER = ["pi_tuned", "pi_sd", "rss_longitudinal"]
WEIGHT_ORDER = ["frequency", "severity"]
DATASET_ORDER = ["ngsim_pooled", "highd_external"]

CONTROLLER_META: Dict[str, Dict[str, object]] = {
    "star": {
        "label": "Upper bound",
        "math_label": "Upper bound",
        "color": "#222222",
        "marker": "o",
        "zorder": 5,
    },
    "pi": {
        "label": "Weak TTC",
        "math_label": "Weak TTC",
        "color": "#4C78A8",
        "marker": "o",
        "zorder": 2,
    },
    "pi_tuned": {
        "label": "Strong TTC",
        "math_label": "Strong TTC",
        "color": "#F58518",
        "marker": "^",
        "zorder": 3,
    },
    "pi_sd": {
        "label": "Stopping-distance",
        "math_label": "Stopping-distance",
        "color": "#54A24B",
        "marker": "s",
        "zorder": 4,
    },
    "rss_longitudinal": {
        "label": "RSS longitudinal",
        "math_label": "RSS longitudinal",
        "color": "#B279A2",
        "marker": "D",
        "zorder": 4,
    },
}

DATASET_META = {
    "ngsim_pooled": {"label": "Pooled NGSIM", "linestyle": "-"},
    "highd_external": {"label": "highD", "linestyle": (0, (5, 3))},
}

WEIGHT_META = {
    "frequency": {"title": "Frequency-weighted coverage"},
    "severity": {"title": "Severity-weighted coverage"},
}

PANEL_TAGS = ["(a)", "(b)", "(c)", "(d)"]


def _controller_column(controller: str) -> str:
    return f"coverage_{controller}"


def _controller_ci_low_column(controller: str) -> str:
    return f"coverage_{controller}_ci_low"


def _controller_ci_high_column(controller: str) -> str:
    return f"coverage_{controller}_ci_high"


# ---------------------------------------------------------------------------
# Result loading layer
# ---------------------------------------------------------------------------
@dataclass
class ResultsStore:
    path: Path
    root_prefix: Optional[str] = None
    zip_obj: Optional[zipfile.ZipFile] = None

    def __post_init__(self) -> None:
        self.path = self.path.expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Results path does not exist: {self.path}")

        if self.path.is_file():
            self.zip_obj = zipfile.ZipFile(self.path)
            self.root_prefix = self._detect_zip_root_prefix(self.zip_obj)
        else:
            self.root_prefix = str(self._detect_directory_root(self.path))

    @staticmethod
    def _detect_zip_root_prefix(zf: zipfile.ZipFile) -> str:
        target_suffix = "ngsim_pooled/coverage_frequency.csv"
        for name in zf.namelist():
            if name.endswith(target_suffix):
                return name[: -len(target_suffix)].rstrip("/")
        raise FileNotFoundError(
            "Could not locate 'ngsim_pooled/coverage_frequency.csv' inside the zip bundle."
        )

    @staticmethod
    def _detect_directory_root(path: Path) -> Path:
        # Case 1: path itself is the results root.
        required = {"ngsim_pooled", "highd_external", "assembled"}
        if required.issubset({p.name for p in path.iterdir() if p.is_dir()}):
            return path
        # Case 2: path contains one results root directory.
        for child in path.iterdir():
            if child.is_dir():
                names = {p.name for p in child.iterdir() if p.is_dir()}
                if required.issubset(names):
                    return child
        raise FileNotFoundError(
            "Could not find an unzipped results root containing assembled/, "
            "ngsim_pooled/, and highd_external/."
        )

    def read_csv(self, relative_path: str) -> pd.DataFrame:
        if self.zip_obj is not None:
            full_name = f"{self.root_prefix}/{relative_path}" if self.root_prefix else relative_path
            try:
                raw = self.zip_obj.read(full_name)
            except KeyError as exc:
                raise FileNotFoundError(f"Missing CSV in zip bundle: {full_name}") from exc
            return pd.read_csv(io.BytesIO(raw))
        full_path = Path(self.root_prefix) / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"Missing CSV in results directory: {full_path}")
        return pd.read_csv(full_path)

    def close(self) -> None:
        if self.zip_obj is not None:
            self.zip_obj.close()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_pdf_and_png(fig: plt.Figure, base_path: Path) -> None:
    ensure_directory(base_path.parent)
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=600, bbox_inches="tight")


def dataset_weight_title(dataset: str, weight_mode: str) -> str:
    return f"{DATASET_META[dataset]['label']} — {weight_mode.capitalize()}"


def tidy_coverage_dataframe(store: ResultsStore) -> pd.DataFrame:
    rows = []
    for dataset in DATASET_ORDER:
        for weight_mode in WEIGHT_ORDER:
            cov = store.read_csv(f"{dataset}/coverage_{weight_mode}.csv").copy()
            for controller in CONTROLLER_ORDER:
                y_col = _controller_column(controller)
                low_col = _controller_ci_low_column(controller)
                high_col = _controller_ci_high_column(controller)
                tmp = pd.DataFrame(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_META[dataset]["label"],
                        "weight_mode": weight_mode,
                        "delta_s": cov["delta_s"],
                        "controller": controller,
                        "controller_label": CONTROLLER_META[controller]["label"],
                        "coverage": cov[y_col],
                        "coverage_pct": cov[y_col] * 100.0,
                        "ci_low": cov[low_col] if low_col in cov.columns else np.nan,
                        "ci_high": cov[high_col] if high_col in cov.columns else np.nan,
                    }
                )
                tmp["ci_low_pct"] = tmp["ci_low"] * 100.0
                tmp["ci_high_pct"] = tmp["ci_high"] * 100.0
                rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def load_main_tables(store: ResultsStore) -> pd.DataFrame:
    frames = []
    for dataset in DATASET_ORDER:
        for weight_mode in WEIGHT_ORDER:
            df = store.read_csv(f"{dataset}/main_table_{weight_mode}.csv").copy()
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["saturation_pct"] = out["saturation"] * 100.0
    out["sat_gap_vs_star_pp"] = out["sat_gap_vs_star"] * 100.0
    return out


def load_stratum_table(store: ResultsStore) -> pd.DataFrame:
    df = store.read_csv("highd_external/location_direction_metric_table.csv").copy()
    df["stratum"] = df.apply(
        lambda r: f"L{int(r['location_id'])}–D{int(r['driving_direction'])}", axis=1
    )
    df["sat_gap_vs_star_pp"] = df["sat_gap_vs_star"] * 100.0
    return df


def quantile_shift_text(main_table: pd.DataFrame, weight_mode: str, controller: str) -> str:
    subset = main_table[
        (main_table["weight_mode"] == weight_mode) & (main_table["controller"] == controller)
    ].set_index("dataset")
    ngsim = subset.loc["ngsim_pooled", "delta_abs_q90_s"]
    highd = subset.loc["highd_external", "delta_abs_q90_s"]
    return f"{ngsim:.2f} → {highd:.2f} s"


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------
def plot_main_transfer_overlay(
    coverage_tidy: pd.DataFrame,
    main_table: pd.DataFrame,
    out_path: Path,
    source_data_path: Path,
    x_max: float = 4.0,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, weight_mode, tag in zip(axes, WEIGHT_ORDER, PANEL_TAGS[:2]):
        panel_df = coverage_tidy[coverage_tidy["weight_mode"] == weight_mode]
        for controller in CONTROLLER_ORDER:
            meta = CONTROLLER_META[controller]
            for dataset in DATASET_ORDER:
                sub = panel_df[
                    (panel_df["dataset"] == dataset) & (panel_df["controller"] == controller)
                ]
                ax.plot(
                    sub["delta_s"],
                    sub["coverage_pct"],
                    color=meta["color"],
                    linestyle=DATASET_META[dataset]["linestyle"],
                    linewidth=1.0 if controller != "star" else 1.2,
                    alpha=0.96,
                    zorder=meta["zorder"],
                )

                # qrow = main_table[
                #     (main_table["dataset"] == dataset)
                #     & (main_table["weight_mode"] == weight_mode)
                #     & (main_table["controller"] == controller)
                # ]
                # if not qrow.empty:
                #     q90 = qrow.iloc[0]["delta_abs_q90_s"]
                #     if pd.notna(q90) and q90 <= x_max + 0.05:
                #         ax.scatter(
                #             [q90],
                #             [90.0],
                #             s=16,
                #             marker=meta["marker"],
                #             facecolor="white",
                #             edgecolor=meta["color"],
                #             linewidth=0.9,
                #             zorder=8,
                #         )

        ax.axhline(90.0, color="0.5", linewidth=0.9, linestyle=(0, (1, 2)), zorder=1)
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, 102.0)
        ax.set_title(WEIGHT_META[weight_mode]["title"])
        ax.set_xlabel("Required lead time, $T_{\\mathrm{req}}$ (s)")
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.text(
            0.01,
            1.02,
            tag,
            transform=ax.transAxes,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        shift_text = (
            f"Upper-bound $\\Delta_{{90}}$: {quantile_shift_text(main_table, weight_mode, 'star')}\n"
            f"RSS $\\Delta_{{90}}$: {quantile_shift_text(main_table, weight_mode, 'rss_longitudinal')}"
        )
        ax.text(
            0.43,
            0.05,
            shift_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.6,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.92, "boxstyle": "round,pad=0.25"},
        )

    axes[0].set_ylabel("Weighted crash coverage (%)")

    controller_handles = [
        Line2D(
            [0],
            [0],
            color=CONTROLLER_META[c]["color"],
            lw=2.0,
            label=CONTROLLER_META[c]["label"],
        )
        for c in CONTROLLER_ORDER
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            lw=1.8,
            linestyle=DATASET_META[d]["linestyle"],
            label=DATASET_META[d]["label"],
        )
        for d in DATASET_ORDER
    ]

    fig.legend(
        handles=controller_handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        columnspacing=1.2,
        handlelength=2.0,
    )
    fig.legend(
        handles=dataset_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.03),
        frameon=False,
        columnspacing=1.5,
        handlelength=2.5,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.20, top=0.83, wspace=0.10)

    save_pdf_and_png(fig, out_path)
    plt.close(fig)

    source = coverage_tidy.copy()
    source.to_csv(source_data_path, index=False)



def plot_supp_transfer_curves(
    coverage_tidy: pd.DataFrame,
    main_table: pd.DataFrame,
    out_path: Path,
    source_data_path: Path,
    x_max: float = 4.0,
    with_ci: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9), sharex=True, sharey=True)
    axes = np.asarray(axes)

    for idx, (dataset, weight_mode) in enumerate(
        [("ngsim_pooled", "frequency"), ("ngsim_pooled", "severity"), ("highd_external", "frequency"), ("highd_external", "severity")]
    ):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        panel_df = coverage_tidy[
            (coverage_tidy["dataset"] == dataset) & (coverage_tidy["weight_mode"] == weight_mode)
        ]

        for controller in CONTROLLER_ORDER:
            meta = CONTROLLER_META[controller]
            sub = panel_df[panel_df["controller"] == controller]
            if with_ci:
                ax.fill_between(
                    sub["delta_s"],
                    sub["ci_low_pct"],
                    sub["ci_high_pct"],
                    color=meta["color"],
                    alpha=0.08,
                    linewidth=0,
                    zorder=1,
                )
            ax.plot(
                sub["delta_s"],
                sub["coverage_pct"],
                color=meta["color"],
                linewidth=1.9 if controller != "star" else 2.2,
                zorder=meta["zorder"],
            )
            qrow = main_table[
                (main_table["dataset"] == dataset)
                & (main_table["weight_mode"] == weight_mode)
                & (main_table["controller"] == controller)
            ]
            if not qrow.empty:
                q90 = qrow.iloc[0]["delta_abs_q90_s"]
                if pd.notna(q90) and q90 <= x_max + 0.05:
                    ax.scatter(
                        [q90],
                        [90.0],
                        s=20,
                        marker=meta["marker"],
                        facecolor="white",
                        edgecolor=meta["color"],
                        linewidth=0.9,
                        zorder=8,
                    )

        ax.axhline(90.0, color="0.55", linewidth=0.9, linestyle=(0, (1, 2)), zorder=1)
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, 102.0)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.set_title(dataset_weight_title(dataset, weight_mode))
        ax.text(
            0.01,
            1.02,
            PANEL_TAGS[idx],
            transform=ax.transAxes,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    for ax in axes[1, :]:
        ax.set_xlabel("Required lead time, $T_{\\mathrm{req}}$ (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Weighted crash coverage (%)")

    controller_handles = [
        Line2D(
            [0],
            [0],
            color=CONTROLLER_META[c]["color"],
            lw=2.0,
            marker=CONTROLLER_META[c]["marker"],
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=CONTROLLER_META[c]["label"],
        )
        for c in CONTROLLER_ORDER
    ]
    fig.legend(
        handles=controller_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=5,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.13, top=0.92, hspace=0.22, wspace=0.10)

    save_pdf_and_png(fig, out_path)
    plt.close(fig)

    coverage_tidy.to_csv(source_data_path, index=False)



# def plot_highd_strata_summary(
#     stratum_df: pd.DataFrame,
#     out_path: Path,
#     source_data_path: Path,
# ) -> None:
#     fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.3), sharey=True)
#     axes = np.atleast_1d(axes)
#     offsets = {"pi_tuned": -0.18, "pi_sd": 0.0, "rss_longitudinal": 0.18}
#     markers = {c: CONTROLLER_META[c]["marker"] for c in STRATUM_CONTROLLER_ORDER}
#
#     ordered_strata = (
#         stratum_df[["location_id", "driving_direction", "stratum"]]
#         .drop_duplicates()
#         .sort_values(["location_id", "driving_direction"])
#         ["stratum"]
#         .tolist()
#     )
#     y_base = np.arange(len(ordered_strata))
#     stratum_to_y = {s: y for y, s in enumerate(ordered_strata)}
#
#     for ax, weight_mode, tag in zip(axes, WEIGHT_ORDER, PANEL_TAGS[:2]):
#         panel = stratum_df[
#             (stratum_df["weight_mode"] == weight_mode)
#             & (stratum_df["controller"].isin(STRATUM_CONTROLLER_ORDER))
#         ].copy()
#         x_max = max(7.0, math.ceil(panel["sat_gap_vs_star_pp"].max() + 0.6))
#
#         for controller in STRATUM_CONTROLLER_ORDER:
#             meta = CONTROLLER_META[controller]
#             sub = panel[panel["controller"] == controller].copy()
#             sub["y"] = sub["stratum"].map(stratum_to_y) + offsets[controller]
#
#             # Faint guide lines from 0 to the point.
#             for _, row in sub.iterrows():
#                 ax.plot(
#                     [0, row["sat_gap_vs_star_pp"]],
#                     [row["y"], row["y"]],
#                     color=meta["color"],
#                     alpha=0.22,
#                     linewidth=1.0,
#                     zorder=1,
#                 )
#
#             ax.scatter(
#                 sub["sat_gap_vs_star_pp"],
#                 sub["y"],
#                 s=34,
#                 color=meta["color"],
#                 marker=markers[controller],
#                 edgecolor="white",
#                 linewidth=0.7,
#                 zorder=3,
#                 label=meta["label"],
#             )
#
#         ax.axvline(0.0, color="0.45", linewidth=0.9, linestyle=(0, (1, 2)), zorder=0)
#         ax.set_xlim(-0.15, x_max)
#         ax.xaxis.set_major_locator(MultipleLocator(1.0))
#         ax.set_title(f"highD strata — {weight_mode.capitalize()}")
#         ax.set_xlabel("Gap to upper bound (percentage points)")
#         ax.text(
#             0.01,
#             1.02,
#             tag,
#             transform=ax.transAxes,
#             fontweight="bold",
#             ha="left",
#             va="bottom",
#         )
#         ax.text(
#             0.98,
#             0.02,
#             "0 pp = tied with upper bound",
#             transform=ax.transAxes,
#             ha="right",
#             va="bottom",
#             fontsize=7.2,
#             color="0.35",
#         )
#
#     axes[0].set_yticks(y_base)
#     axes[0].set_yticklabels(ordered_strata)
#     axes[0].invert_yaxis()
#     axes[0].set_ylabel("Location-direction stratum")
#     axes[1].invert_yaxis()
#
#     handles = [
#         Line2D(
#             [0],
#             [0],
#             color=CONTROLLER_META[c]["color"],
#             lw=1.2,
#             marker=CONTROLLER_META[c]["marker"],
#             markersize=5.5,
#             label=CONTROLLER_META[c]["label"],
#         )
#         for c in STRATUM_CONTROLLER_ORDER
#     ]
#     fig.legend(
#         handles=handles,
#         loc="lower center",
#         bbox_to_anchor=(0.5, -0.01),
#         ncol=3,
#         frameon=False,
#         handlelength=2.0,
#         columnspacing=1.8,
#     )
#     fig.subplots_adjust(left=0.17, right=0.995, bottom=0.12, top=0.90, wspace=0.08)
#
#     save_pdf_and_png(fig, out_path)
#     plt.close(fig)
#
#     stratum_df.to_csv(source_data_path, index=False)
# def plot_highd_strata_summary(
#     stratum_df: pd.DataFrame,
#     out_path: Path,
#     source_data_path: Path,
# ) -> None:
#     from matplotlib.patches import Patch
#
#     ordered_meta = (
#         stratum_df[["location_id", "driving_direction", "stratum"]]
#         .drop_duplicates()
#         .sort_values(["location_id", "driving_direction"])
#         .reset_index(drop=True)
#     )
#     ordered_strata = ordered_meta["stratum"].tolist()
#     n_strata = len(ordered_strata)
#
#     fig_height = max(4.2, 0.30 * n_strata + 1.2)
#     fig, axes = plt.subplots(1, 2, figsize=(7.2, fig_height), sharey=True)
#     axes = np.atleast_1d(axes)
#
#     bar_height = 0.22
#     offsets = {
#         "pi_tuned": -bar_height,
#         "pi_sd": 0.0,
#         "rss_longitudinal": bar_height,
#     }
#     panel_titles = {
#         "frequency": "Frequency weighting",
#         "severity": "Severity weighting",
#     }
#
#     y_base = np.arange(n_strata)
#     stratum_to_y = dict(zip(ordered_strata, y_base))
#
#     # Add subtle location_id grouping bands without drawing visual focus.
#     group_bounds = []
#     start = 0
#     location_ids = ordered_meta["location_id"].tolist()
#     while start < len(location_ids):
#         end = start
#         while end + 1 < len(location_ids) and location_ids[end + 1] == location_ids[start]:
#             end += 1
#         group_bounds.append((start, end))
#         start = end + 1
#
#     # Share x-axis limits across both panels for horizontal comparison.
#     finite_vals = (
#         stratum_df[stratum_df["controller"].isin(STRATUM_CONTROLLER_ORDER)]["sat_gap_vs_star_pp"]
#         .replace([np.inf, -np.inf], np.nan)
#         .dropna()
#     )
#     if finite_vals.empty:
#         x_lower, x_upper = 0.0, 4.0
#     else:
#         x_min = float(finite_vals.min())
#         x_max = float(finite_vals.max())
#         x_lower = min(0.0, math.floor((x_min - 0.25) * 2.0) / 2.0)
#         x_upper = max(4.0, math.ceil((x_max + 0.35) * 2.0) / 2.0)
#
#     for ax, weight_mode, tag in zip(axes, WEIGHT_ORDER, PANEL_TAGS[:2]):
#         panel = stratum_df[
#             (stratum_df["weight_mode"] == weight_mode)
#             & (stratum_df["controller"].isin(STRATUM_CONTROLLER_ORDER))
#         ].copy()
#         panel["y0"] = panel["stratum"].map(stratum_to_y)
#
#         ax.set_axisbelow(True)
#         ax.grid(False, axis="y")
#         ax.grid(True, axis="x", color="0.25", alpha=0.16, linewidth=0.5)
#
#         # Very light grouping band plus a thin divider.
#         for group_idx, (start_idx, end_idx) in enumerate(group_bounds):
#             if group_idx % 2 == 0:
#                 ax.axhspan(start_idx - 0.5, end_idx + 0.5, color="0.985", zorder=0)
#             if end_idx < n_strata - 1:
#                 ax.axhline(end_idx + 0.5, color="0.90", linewidth=0.7, zorder=1)
#
#         # 0 pp reference line.
#         ax.axvline(0.0, color="0.45", linewidth=0.9, linestyle=(0, (1, 2)), zorder=2)
#
#         # Grouped horizontal bars.
#         for controller in STRATUM_CONTROLLER_ORDER:
#             meta = CONTROLLER_META[controller]
#             sub = panel[panel["controller"] == controller].copy()
#             sub["y"] = sub["y0"] + offsets[controller]
#
#             ax.barh(
#                 sub["y"],
#                 sub["sat_gap_vs_star_pp"],
#                 height=bar_height * 0.90,
#                 color=meta["color"],
#                 edgecolor="white",
#                 linewidth=0.6,
#                 alpha=0.95,
#                 zorder=3,
#                 label=meta["label"],
#             )
#
#         ax.set_xlim(x_lower, x_upper)
#         ax.set_ylim(n_strata - 0.5, -0.5)
#         ax.xaxis.set_major_locator(MultipleLocator(1.0 if x_upper >= 5.0 else 0.5))
#         ax.tick_params(axis="y", length=0)
#         ax.set_title(panel_titles.get(weight_mode, weight_mode.capitalize()), pad=8.0)
#         ax.set_xlabel("Saturation gap to upper bound (pp)")
#         ax.text(
#             0.01,
#             1.02,
#             tag,
#             transform=ax.transAxes,
#             fontweight="bold",
#             ha="left",
#             va="bottom",
#         )
#         ax.text(
#             0.98,
#             0.02,
#             "0 pp = tied with upper bound",
#             transform=ax.transAxes,
#             ha="right",
#             va="bottom",
#             fontsize=7.2,
#             color="0.35",
#         )
#
#     axes[0].set_yticks(y_base)
#     axes[0].set_yticklabels(ordered_strata)
#     axes[0].set_ylabel("Location-direction stratum")
#     axes[1].tick_params(axis="y", labelleft=False, length=0)
#
#     handles = [
#         Patch(
#             facecolor=CONTROLLER_META[c]["color"],
#             edgecolor="white",
#             label=CONTROLLER_META[c]["label"],
#         )
#         for c in STRATUM_CONTROLLER_ORDER
#     ]
#     fig.legend(
#         handles=handles,
#         loc="upper center",
#         bbox_to_anchor=(0.5, 0.995),
#         ncol=3,
#         frameon=False,
#         columnspacing=1.6,
#         handletextpad=0.6,
#     )
#
#     fig.subplots_adjust(left=0.18, right=0.995, bottom=0.11, top=0.87, wspace=0.08)
#
#     save_pdf_and_png(fig, out_path)
#     plt.close(fig)
#
#     stratum_df.to_csv(source_data_path, index=False)

def plot_highd_strata_summary(
    stratum_df: pd.DataFrame,
    out_path: Path,
    source_data_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.7), sharey=True)
    axes = np.atleast_1d(axes)

    offsets = {"pi_tuned": -0.16, "pi_sd": 0.0, "rss_longitudinal": 0.16}
    panel_titles = {
        "frequency": "Frequency weighting",
        "severity": "Severity weighting",
    }

    ordered_meta = (
        stratum_df[["location_id", "driving_direction", "stratum"]]
        .drop_duplicates()
        .sort_values(["location_id", "driving_direction"])
        .reset_index(drop=True)
    )
    ordered_strata = ordered_meta["stratum"].tolist()
    y_base = np.arange(len(ordered_strata))
    stratum_to_y = dict(zip(ordered_strata, y_base))

    # Group rows by location_id so that each location pair gets a subtle band.
    group_bounds = []
    start = 0
    location_ids = ordered_meta["location_id"].tolist()
    while start < len(location_ids):
        end = start
        while end + 1 < len(location_ids) and location_ids[end + 1] == location_ids[start]:
            end += 1
        group_bounds.append((start, end))
        start = end + 1

    # Use one common x-range across both panels for direct comparison.
    finite_vals = (
        stratum_df[stratum_df["controller"].isin(STRATUM_CONTROLLER_ORDER)]["sat_gap_vs_star_pp"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if finite_vals.empty:
        x_lower, x_upper = -0.1, 4.0
    else:
        x_min = float(finite_vals.min())
        x_max = float(finite_vals.max())
        x_lower = -0.1 if x_min >= 0.0 else math.floor((x_min - 0.25) * 2.0) / 2.0
        x_upper = max(4.0, math.ceil((x_max + 0.35) * 2.0) / 2.0)

    for ax, weight_mode, tag in zip(axes, WEIGHT_ORDER, PANEL_TAGS[:2]):
        panel = stratum_df[
            (stratum_df["weight_mode"] == weight_mode)
            & (stratum_df["controller"].isin(STRATUM_CONTROLLER_ORDER))
        ].copy()
        panel["y0"] = panel["stratum"].map(stratum_to_y)

        # Cleaner grid: x-grid only.
        ax.set_axisbelow(True)
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", color="0.25", alpha=0.18, linewidth=0.5)

        # Very light alternating location bands and thin separators.
        for group_idx, (start_idx, end_idx) in enumerate(group_bounds):
            if group_idx % 2 == 0:
                ax.axhspan(start_idx - 0.5, end_idx + 0.5, color="0.975", zorder=0)
            if end_idx < len(ordered_strata) - 1:
                ax.axhline(end_idx + 0.5, color="0.88", linewidth=0.7, zorder=0)

        # Reference line: tied with upper bound.
        ax.axvline(0.0, color="0.45", linewidth=0.9, linestyle=(0, (1, 2)), zorder=1)

        # Neutral stems + restrained hollow markers.
        for controller in STRATUM_CONTROLLER_ORDER:
            meta = CONTROLLER_META[controller]
            sub = panel[panel["controller"] == controller].copy()
            sub["y"] = sub["y0"] + offsets[controller]

            x_values = sub["sat_gap_vs_star_pp"].to_numpy()
            y_values = sub["y"].to_numpy()

            ax.hlines(
                y=y_values,
                xmin=np.minimum(0.0, x_values),
                xmax=np.maximum(0.0, x_values),
                color="0.80",
                linewidth=0.8,
                zorder=1,
            )
            ax.scatter(
                x_values,
                y_values,
                s=34,
                marker=meta["marker"],
                facecolor="white",
                edgecolor=meta["color"],
                linewidth=1.0,
                alpha=0.98,
                zorder=3,
                label=meta["label"],
            )

        ax.set_xlim(x_lower, x_upper)
        ax.set_ylim(len(ordered_strata) - 0.5, -0.5)
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.tick_params(axis="y", length=0)
        ax.set_title(panel_titles.get(weight_mode, weight_mode.capitalize()), pad=8.0)
        ax.set_xlabel("Saturation gap to upper bound (pp)")
        ax.text(
            0.01,
            1.02,
            tag,
            transform=ax.transAxes,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    axes[0].set_yticks(y_base)
    axes[0].set_yticklabels(ordered_strata)
    axes[0].set_ylabel("Location-direction stratum")
    axes[1].tick_params(labelleft=False)

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker=CONTROLLER_META[c]["marker"],
            markersize=5.8,
            markerfacecolor="white",
            markeredgecolor=CONTROLLER_META[c]["color"],
            markeredgewidth=1.0,
            label=CONTROLLER_META[c]["label"],
        )
        for c in STRATUM_CONTROLLER_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        columnspacing=1.6,
        handletextpad=0.5,
    )

    fig.subplots_adjust(left=0.18, right=0.995, bottom=0.13, top=0.86, wspace=0.08)

    save_pdf_and_png(fig, out_path)
    plt.close(fig)

    stratum_df.to_csv(source_data_path, index=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RSS/highD transfer figures for the LaTeX paper."
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to rss_highd_zero_tune_results_v1.0.zip or the unzipped rss_highd_zero_tune directory.",
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=Path("."),
        help="Paper project root containing paper_figures/. Defaults to the current directory.",
    )
    parser.add_argument(
        "--xmax-main",
        type=float,
        default=4.0,
        help="Maximum x-axis limit for the main-text overlay figure.",
    )
    parser.add_argument(
        "--xmax-supp",
        type=float,
        default=4.0,
        help="Maximum x-axis limit for the supplementary transfer-curves figure.",
    )
    parser.add_argument(
        "--with-ci",
        action="store_true",
        help="Add very light bootstrap confidence ribbons to the supplementary transfer-curves figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_root = args.paper_root.expanduser().resolve()

    main_dir = paper_root / "paper_figures" / "main"
    supp_dir = paper_root / "paper_figures" / "supplement"
    source_dir = paper_root / "paper_figures" / "source_data"
    for directory in (main_dir, supp_dir, source_dir):
        ensure_directory(directory)

    store = ResultsStore(args.results)
    try:
        coverage_tidy = tidy_coverage_dataframe(store)
        main_table = load_main_tables(store)
        stratum_df = load_stratum_table(store)

        plot_main_transfer_overlay(
            coverage_tidy=coverage_tidy,
            main_table=main_table,
            out_path=main_dir / "fig05_main_rss_highd_transfer",
            source_data_path=source_dir / "fig05_main_rss_highd_transfer.csv",
            x_max=args.xmax_main,
        )
        plot_supp_transfer_curves(
            coverage_tidy=coverage_tidy,
            main_table=main_table,
            out_path=supp_dir / "figS08_supp_rss_highd_curves",
            source_data_path=source_dir / "figS08_supp_rss_highd_curves.csv",
            x_max=args.xmax_supp,
            with_ci=args.with_ci,
        )
        plot_highd_strata_summary(
            stratum_df=stratum_df,
            out_path=supp_dir / "figS09_supp_highd_strata",
            source_data_path=source_dir / "figS09_supp_highd_strata.csv",
        )
    finally:
        store.close()

    print("Generated figure files:")
    print(f"  {main_dir / 'fig05_main_rss_highd_transfer.pdf'}")
    print(f"  {main_dir / 'fig05_main_rss_highd_transfer.png'}")
    print(f"  {supp_dir / 'figS08_supp_rss_highd_curves.pdf'}")
    print(f"  {supp_dir / 'figS08_supp_rss_highd_curves.png'}")
    print(f"  {supp_dir / 'figS09_supp_highd_strata.pdf'}")
    print(f"  {supp_dir / 'figS09_supp_highd_strata.png'}")


if __name__ == "__main__":
    main()
