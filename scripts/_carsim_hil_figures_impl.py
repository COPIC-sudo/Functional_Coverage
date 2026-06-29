from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, PercentFormatter


PAPER_FIGURE_SPECS = {
    "figCH01_main_carsim_validation_summary": "Main text · CarSim side-consistency, boundary-error, impact-speed, and ranking summaries.",
    "figCH02_main_hil_reproducibility_summary": "Main text · HIL timing and reproducibility summaries against acceptance targets.",
    "figCHS01_supp_carsim_boundary_diagnostics": "Supplement · CarSim boundary-transfer diagnostics (bracketing, error-share, affine-fit summaries).",
    "figCHS02_supp_carsim_family_cell_breakdown": "Supplement · CarSim family, uncertainty-cell, and interaction breakdowns.",
    "figCHS03_supp_carsim_pairwise_ranking": "Supplement · CarSim pairwise ranking gaps and preservation rates.",
    "figCHS04_supp_hil_design_subset": "Supplement · HIL subset coverage and composition diagnostics.",
}


@dataclass(frozen=True)
class FigureOutput:
    stem: str
    pdf_path: Path
    png_path: Path
    source_data_path: Path | None
    description: str


SCOPE_ORDER = ["US-101 prototypes", "I-80 prototypes", "Combined"]
SCOPE_LABELS = {
    "US-101 prototypes": "US-101",
    "I-80 prototypes": "I-80",
    "Combined": "Combined",
}

SLICE_TYPE_ORDER = ["family", "uncertainty_cell", "interaction"]
SLICE_TYPE_LABELS = {
    "family": "Family",
    "uncertainty_cell": "Uncertainty cell",
    "interaction": "Interaction",
}
SLICE_TYPE_COLORS = {
    "family": "#4C78A8",
    "uncertainty_cell": "#F58518",
    "interaction": "#9C755F",
}

PAIR_LABELS = {
    "star_vs_pi_tuned": "Upper bound vs Strong TTC",
    "star_vs_pi_sd": "Upper bound vs Stopping-distance",
    "pi_vs_pi_tuned": "Weak TTC vs Strong TTC",
    "pi_vs_pi_sd": "Weak TTC vs Stopping-distance",
    "pi_tuned_vs_pi_sd": "Strong TTC vs Stopping-distance",
}

PAIR_COLORS = {
    "star_vs_pi_tuned": "#111111",
    "star_vs_pi_sd": "#111111",
    "pi_vs_pi_tuned": "#8A8F98",
    "pi_vs_pi_sd": "#8A8F98",
    "pi_tuned_vs_pi_sd": "#9467BD",
}

HIL_COLORS = {
    "timing": "#4C78A8",
    "agreement": "#54A24B",
    "rare": "#F58518",
    "diagnostic": "#9C755F",
}


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
            "lines.linewidth": 1.8,
            "grid.alpha": 0.20,
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


def _resolve_results_root(results_root: str | Path) -> Path:
    path = Path(results_root).expanduser().resolve()
    candidates = [
        path,
        path / "carsim_hil_results_csv",
        path / "results" / "carsim_hil_results_csv",
    ]
    required = {f"0{i}_" for i in range(1, 8)}
    for candidate in candidates:
        if candidate.exists():
            names = {p.name[:3] for p in candidate.glob("*.csv")}
            if required.issubset(names):
                return candidate
            if (candidate / "01_carsim_main_table.csv").exists() and (candidate / "05_hil_main_table.csv").exists():
                return candidate
    tried = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate the CarSim/HIL results folder. Tried:\n"
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


def _source_and_save(
    fig: plt.Figure,
    stem: str,
    target_dir: Path,
    source_df: pd.DataFrame | None,
    source_dir: Path,
    png_dpi: int,
) -> FigureOutput:
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


def _format_scope(scope: str) -> str:
    return SCOPE_LABELS.get(scope, scope)


def _fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _fmt_num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _make_carsim_summary_figure(df: pd.DataFrame, main_dir: Path, source_dir: Path, png_dpi: int) -> FigureOutput:
    data = df.copy()
    data["Scope"] = pd.Categorical(data["Scope"], categories=SCOPE_ORDER, ordered=True)
    data = data.sort_values("Scope")
    scopes = [SCOPE_LABELS[str(s)] for s in data["Scope"]]
    x = np.arange(len(scopes))
    width = 0.22

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.2), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel A: side consistency
    side_metrics = [
        ("p_succ^+", r"$p_{succ}^{+}$", "#4C78A8"),
        ("p_crash^-", r"$p_{crash}^{-}$", "#F58518"),
        ("A_side", r"$A_{side}$", "#111111"),
    ]
    offsets = [-width, 0.0, width]
    label_offsets = {
        "p_succ^+": 0.008,
        "p_crash^-": 0.016,
        "A_side": 0.024,
    }

    for (col, label, color), dx in zip(side_metrics, offsets):
        vals = data[col].to_numpy(dtype=float)
        bars = ax1.bar(x + dx, vals, width=width, color=color, label=label)
        dy = label_offsets[col]
        for bar, v in zip(bars, vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                min(v + dy, 1.005),
                _fmt_pct(v),
                ha="center",
                va="bottom",
                fontsize=6,
            )

    ax1.axhline(0.85, color="#808080", linestyle=(0, (3, 2)), linewidth=1.0)
    ax1.axhline(0.90, color="#B0B0B0", linestyle=(0, (1, 2)), linewidth=1.0)
    ax1.text(len(scopes) - 0.35, 0.853, "0.85", fontsize=7, color="#707070", va="bottom")
    ax1.text(len(scopes) - 0.35, 0.903, "0.90", fontsize=7, color="#8A8A8A", va="bottom")
    ax1.set_ylim(0, 1.07)
    ax1.set_xticks(x, scopes)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_ylabel("Rate")
    ax1.set_title("(a) Side-consistency metrics", pad=18)
    ax1.grid(axis="y")
    ax1.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0.0,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    _despine(ax1)

    # Panel B: boundary timing error
    timing_metrics = [
        ("Median |e_Δ| (s)", r"Median $|e_{\Delta}|$", "#4C78A8"),
        ("P90 |e_Δ| (s)", r"P90 $|e_{\Delta}|$", "#F58518"),
    ]
    for (col, label, color), dx in zip(timing_metrics, [-width / 2, width / 2]):
        vals = data[col].to_numpy(dtype=float)
        bars = ax2.bar(x + dx, vals, width=width, color=color, label=label)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.008, _fmt_num(v, 2), ha="center", va="bottom", fontsize=7)
    ax2.axhline(0.15, color="#808080", linestyle=(0, (3, 2)), linewidth=1.0)
    ax2.axhline(0.25, color="#B0B0B0", linestyle=(0, (1, 2)), linewidth=1.0)
    ax2.text(len(scopes) - 0.35, 0.153, "0.15 s", fontsize=7, color="#707070", va="bottom")
    ax2.text(len(scopes) - 0.35, 0.253, "0.25 s", fontsize=7, color="#8A8A8A", va="bottom")
    ax2.set_ylim(0, max(0.32, float(data[["Median |e_Δ| (s)", "P90 |e_Δ| (s)"]].max().max()) + 0.04))
    ax2.set_xticks(x, scopes)
    ax2.set_ylabel("Seconds")
    ax2.set_title("(b) Boundary-transfer timing error")
    ax2.grid(axis="y")
    ax2.legend(loc="upper left")
    _despine(ax2)

    # Panel C: impact-speed error
    impact_vals = data["Median |e_v| (m/s)"].to_numpy(dtype=float)
    bars = ax3.bar(x, impact_vals, width=0.50, color="#9C755F")
    for bar, v in zip(bars, impact_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.05, _fmt_num(v, 2), ha="center", va="bottom", fontsize=7)
    ax3.axhline(1.5, color="#808080", linestyle=(0, (3, 2)), linewidth=1.0)
    ax3.text(len(scopes) - 0.35, 1.53, "1.5 m/s", fontsize=7, color="#707070", va="bottom")
    ax3.set_ylim(0, max(1.75, float(data["Median |e_v| (m/s)"].max()) + 0.25))
    ax3.set_xticks(x, scopes)
    ax3.set_ylabel("m/s")
    ax3.set_title("(c) Impact-speed error")
    ax3.grid(axis="y")
    _despine(ax3)

    # Panel D: ranking preservation
    ranking_vals = data["Ranking Preserved (%)"].to_numpy(dtype=float) / 100.0
    bars = ax4.bar(x, ranking_vals, width=0.50, color="#54A24B")
    for bar, v in zip(bars, ranking_vals):
        ax4.text(bar.get_x() + bar.get_width() / 2, v + 0.012, _fmt_pct(v), ha="center", va="bottom", fontsize=7)
    ax4.axhline(0.90, color="#808080", linestyle=(0, (3, 2)), linewidth=1.0)
    ax4.text(len(scopes) - 0.35, 0.903, "0.90", fontsize=7, color="#707070", va="bottom")
    ax4.set_ylim(0, 1.02)
    ax4.set_xticks(x, scopes)
    ax4.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax4.set_ylabel("Rate")
    ax4.set_title("(d) Ranking preservation")
    ax4.grid(axis="y")
    _despine(ax4)

    source_df = data.copy()
    source_df["Scope"] = source_df["Scope"].astype(str)
    return _source_and_save(fig, "figCH01_main_carsim_validation_summary", main_dir, source_df, source_dir, png_dpi)


def _make_hil_summary_figure(main_df: pd.DataFrame, main_dir: Path, source_dir: Path, png_dpi: int) -> FigureOutput:
    df = main_df.copy()

    label_map = {
        "deadline_miss_rate": "Deadline miss rate",
        "unmatched_trigger_rate": "Unmatched trigger rate",
        "p95_end_to_end_latency_ms": "P95 end-to-end latency",
        "median_abs_e_trig_ms": r"Median $|e_{trig}|$",
        "p95_abs_e_trig_ms": r"P95 $|e_{trig}|$",
        "p95_loop_jitter_ms": "P95 loop jitter",
        "outcome_agreement_vs_offline": "Agreement vs offline",
        "outcome_agreement_vs_carsim": "Agreement vs CarSim",
        "ranking_preservation": "Ranking preservation",
    }
    target_map = {
        "deadline_miss_rate": 0.005,
        "p95_end_to_end_latency_ms": 50.0,
        "median_abs_e_trig_ms": 50.0,
        "p95_abs_e_trig_ms": 100.0,
        "outcome_agreement_vs_offline": 0.90,
        "outcome_agreement_vs_carsim": 0.90,
        "ranking_preservation": 0.90,
    }

    df["Label"] = df["Metric Key"].map(label_map).fillna(df["Metric Label"])
    df["Target"] = df["Metric Key"].map(target_map)

    rare = df[df["Metric Key"].isin(["deadline_miss_rate", "unmatched_trigger_rate"])].copy()
    latency = df[df["Metric Key"].isin(["p95_end_to_end_latency_ms", "median_abs_e_trig_ms", "p95_abs_e_trig_ms", "p95_loop_jitter_ms"])].copy()
    agreement = df[df["Metric Key"].isin(["outcome_agreement_vs_offline", "outcome_agreement_vs_carsim", "ranking_preservation"])].copy()

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.5), constrained_layout=True)
    ax1, ax2, ax3 = axes

    # Rare-event fractions
    rare = rare.iloc[::-1].copy()
    rare_vals = rare["Reported Value"].to_numpy(dtype=float) * 100.0
    y = np.arange(len(rare))
    bars = ax1.barh(y, rare_vals, color=HIL_COLORS["rare"], height=0.58)
    for yi, bar, (_, row) in zip(y, bars, rare.iterrows()):
        ax1.text(bar.get_width() + 0.04, yi, f"{row['Reported Value'] * 100:.2f}%", va="center", fontsize=7)
    miss_target = float(target_map["deadline_miss_rate"] * 100.0)
    ax1.axvline(miss_target, color="#808080", linestyle=(0, (3, 2)), linewidth=1.0)
    ax1.text(miss_target + 0.05, -0.42, "deadline target", fontsize=7, color="#707070")
    ax1.set_yticks(y, rare["Label"])
    ax1.set_xlabel("Percent of runs")
    ax1.set_title("(a) Rare-event timing faults")
    ax1.grid(axis="x")
    ax1.set_xlim(0, max(0.8, float(rare_vals.max()) * 1.55))
    _despine(ax1)

    # Latency metrics with target markers
    latency = latency.iloc[::-1].copy()
    lat_vals = latency["Reported Value"].to_numpy(dtype=float)
    y = np.arange(len(latency))
    bars = ax2.barh(y, lat_vals, color=HIL_COLORS["timing"], height=0.58)
    for yi, bar, (_, row) in zip(y, bars, latency.iterrows()):
        ax2.text(bar.get_width() + 2.5, yi, f"{row['Reported Value']:.0f} ms", va="center", fontsize=7)
        if pd.notna(row["Target"]):
            ax2.plot([float(row["Target"])], [yi], marker="|", markersize=14, color="#111111", markeredgewidth=1.4)
    ax2.set_yticks(y, latency["Label"])
    ax2.set_xlabel("Milliseconds")
    ax2.set_title("(b) End-to-end latency and trigger error")
    ax2.grid(axis="x")
    ax2.set_xlim(0, max(120.0, float(lat_vals.max()) * 1.40))
    ax2.text(ax2.get_xlim()[1] * 0.72, -0.42, "black marker = acceptance target", fontsize=7, color="#444444")
    _despine(ax2)

    # Agreements / preservation
    agreement = agreement.iloc[::-1].copy()
    agr_vals = agreement["Reported Value"].to_numpy(dtype=float)
    y = np.arange(len(agreement))
    bars = ax3.barh(y, agr_vals, color=HIL_COLORS["agreement"], height=0.58)
    for yi, bar, (_, row) in zip(y, bars, agreement.iterrows()):
        ax3.text(min(bar.get_width() + 0.008, 0.992), yi, _fmt_pct(float(row["Reported Value"])), va="center", fontsize=7)
    ax3.axvline(0.90, color="#808080", linestyle=(0, (3, 2)), linewidth=1.0)
    ax3.text(0.902, -0.42, "0.90 target", fontsize=7, color="#707070")
    ax3.set_yticks(y, agreement["Label"])
    ax3.set_xlabel("Rate")
    ax3.set_xlim(0, 1.0)
    ax3.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax3.set_title("(c) Outcome agreement and ranking")
    ax3.grid(axis="x")
    _despine(ax3)

    source_df = df[["Metric Key", "Label", "Unit", "Reported Value", "Target"]].copy()
    return _source_and_save(fig, "figCH02_main_hil_reproducibility_summary", main_dir, source_df, source_dir, png_dpi)


def _make_carsim_diagnostics_figure(df: pd.DataFrame, supp_dir: Path, source_dir: Path, png_dpi: int) -> FigureOutput:
    data = df.copy()
    data["Scope"] = pd.Categorical(data["Scope"], categories=SCOPE_ORDER, ordered=True)
    data = data.sort_values("Scope")
    scopes = [SCOPE_LABELS[str(s)] for s in data["Scope"]]
    x = np.arange(len(scopes))
    width = 0.22

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.2), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel A: bracket + error shares
    cols = [
        ("Boundary Bracket Rate", "Bracketed", "#4C78A8"),
        ("Share |e_Δ| ≤ 0.15 s", r"Share $|e_{\Delta}|\leq0.15$ s", "#72B7B2"),
        ("Share |e_Δ| ≤ 0.25 s", r"Share $|e_{\Delta}|\leq0.25$ s", "#54A24B"),
    ]
    for (col, label, color), dx in zip(cols, [-width, 0.0, width]):
        vals = data[col].to_numpy(dtype=float)
        bars = ax1.bar(x + dx, vals, width=width, color=color, label=label)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.012, _fmt_pct(v), ha="center", va="bottom", fontsize=7)
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks(x, scopes)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_ylabel("Rate")
    ax1.set_title("(a) Boundary bracketing and error-share diagnostics", pad=18)
    ax1.grid(axis="y")
    ax1.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0.0,
        ncol=3,
        columnspacing=1.1,
        handletextpad=0.6,
    )
    _despine(ax1)

    # Panel B: non-monotone + R2
    nonmono = data["Non-monotone Rate"].to_numpy(dtype=float)
    r2 = data["R²"].to_numpy(dtype=float)
    bars1 = ax2.bar(x - width / 2, r2, width=width, color="#F58518", label=r"$R^2$")
    bars2 = ax2.bar(x + width / 2, nonmono, width=width, color="#E45756", label="Non-monotone")
    for bar, v in zip(bars1, r2):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.012, _fmt_num(v, 2), ha="center", va="bottom", fontsize=7)
    for bar, v in zip(bars2, nonmono):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.012, _fmt_pct(v), ha="center", va="bottom", fontsize=7)
    ax2.set_ylim(0, 1.02)
    ax2.set_xticks(x, scopes)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_ylabel("Rate")
    ax2.set_title("(b) Monotonicity and affine-fit quality", pad=18)
    ax2.grid(axis="y")
    ax2.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0.0,
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.6,
    )
    _despine(ax2)

    # Panel C: slope proximity to identity
    slope = data["Scatter Slope"].to_numpy(dtype=float)
    bars = ax3.bar(x, slope, width=0.50, color="#9C755F")
    for bar, v in zip(bars, slope):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.008, _fmt_num(v, 2), ha="center", va="bottom", fontsize=7)
    ax3.axhline(1.0, color="#808080", linestyle=(0, (3, 2)), linewidth=1.0)
    ax3.text(len(scopes) - 0.30, 1.003, "identity", fontsize=7, color="#707070", va="bottom")
    ax3.set_ylim(0.90, 1.08)
    ax3.set_xticks(x, scopes)
    ax3.set_ylabel("Slope")
    ax3.set_title("(c) Predicted-vs-observed affine slope")
    ax3.grid(axis="y")
    _despine(ax3)

    # Panel D: intercept + sample counts
    intercept = data["Intercept (s)"].to_numpy(dtype=float)
    bars = ax4.bar(x, intercept, width=0.50, color="#BAB0AC")
    for bar, v, colliding, groups in zip(
        bars,
        intercept,
        data["Colliding Runs for |e_v|"].to_numpy(dtype=float),
        data["Bracketed Groups for |e_Δ|"].to_numpy(dtype=float),
    ):
        y_text = v + 0.008 if v >= 0 else v - 0.012
        va = "bottom" if v >= 0 else "top"
        ax4.text(bar.get_x() + bar.get_width() / 2, y_text, f"{v:+.2f} s", ha="center", va=va, fontsize=7)
        ax4.text(bar.get_x() + bar.get_width() / 2, -0.043, f"n={int(groups)}/{int(colliding)}", ha="center", va="top", fontsize=7, color="#555555")
    ax4.axhline(0.0, color="#808080", linestyle=(0, (3, 2)), linewidth=1.0)
    ax4.set_ylim(-0.055, 0.055)
    ax4.set_xticks(x, scopes)
    ax4.set_ylabel("Seconds")
    ax4.set_title("(d) Intercept and usable case counts")
    ax4.grid(axis="y")
    _despine(ax4)

    source_df = data.copy()
    source_df["Scope"] = source_df["Scope"].astype(str)
    return _source_and_save(fig, "figCHS01_supp_carsim_boundary_diagnostics", supp_dir, source_df, source_dir, png_dpi)


def _label_slice(row: pd.Series) -> str:
    slice_name = str(row["Slice Name"])
    return slice_name.replace("_", " ")


def _make_family_cell_figure(df: pd.DataFrame, supp_dir: Path, source_dir: Path, png_dpi: int) -> FigureOutput:
    data = df.copy()
    data["Slice Type"] = pd.Categorical(data["Slice Type"], categories=SLICE_TYPE_ORDER, ordered=True)
    data = data.sort_values(["Slice Type", "Slice Name"]).reset_index(drop=True)
    data["Display"] = data.apply(_label_slice, axis=1)
    y = np.arange(len(data))
    colors = [SLICE_TYPE_COLORS.get(str(t), "#777777") for t in data["Slice Type"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), constrained_layout=True)
    ax1, ax2 = axes

    # A_side
    bars = ax1.barh(y, data["A_side"].to_numpy(dtype=float), color=colors, height=0.65)
    for yi, bar, v in zip(y, bars, data["A_side"].to_numpy(dtype=float)):
        ax1.text(min(v + 0.01, 0.995), yi, _fmt_pct(v), va="center", fontsize=7)
    ax1.set_yticks(y, data["Display"])
    ax1.set_xlim(0, 1.0)
    ax1.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_xlabel(r"$A_{side}$")
    ax1.set_title("(a) Side-consistency by family / cell / interaction", pad=18)
    ax1.grid(axis="x")
    _despine(ax1)

    # Error metrics
    med = data["Median |e_Δ| (s)"].to_numpy(dtype=float)
    p90 = data["P90 |e_Δ| (s)"].to_numpy(dtype=float)
    ax2.barh(y + 0.18, med, color="#4C78A8", height=0.30, label=r"Median $|e_{\Delta}|$")
    valid = np.isfinite(p90)
    ax2.barh(y[valid] - 0.18, p90[valid], color="#F58518", height=0.30, label=r"P90 $|e_{\Delta}|$")
    for yi, v in zip(y + 0.18, med):
        ax2.text(v + 0.008, yi, _fmt_num(v, 2), va="center", fontsize=7)
    for yi, v in zip(y[valid] - 0.18, p90[valid]):
        ax2.text(v + 0.008, yi, _fmt_num(v, 2), va="center", fontsize=7)
    for yi, v in zip(y[~valid] - 0.18, p90[~valid]):
        ax2.text(0.01, yi, "NA", va="center", fontsize=7, color="#666666")
    ax2.set_yticks(y, data["Display"])
    ax2.set_xlim(0, max(0.34, np.nanmax(np.r_[med, p90]) + 0.05))
    ax2.set_xlabel("Seconds")
    ax2.set_title("(b) Boundary-transfer error by slice", pad=18)
    ax2.grid(axis="x")
    ax2.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0.0,
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.6,
    )
    _despine(ax2)

    # Type legend
    handles = [Rectangle((0, 0), 1, 1, color=SLICE_TYPE_COLORS[k]) for k in SLICE_TYPE_ORDER]
    labels = [SLICE_TYPE_LABELS[k] for k in SLICE_TYPE_ORDER]
    ax1.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0.0,
        ncol=3,
        columnspacing=1.1,
        handletextpad=0.6,
    )

    source_df = data[["Slice Type", "Slice Name", "A_side", "Median |e_Δ| (s)", "P90 |e_Δ| (s)"]].copy()
    return _source_and_save(fig, "figCHS02_supp_carsim_family_cell_breakdown", supp_dir, source_df, source_dir, png_dpi)


def _make_pairwise_ranking_figure(df: pd.DataFrame, supp_dir: Path, source_dir: Path, png_dpi: int) -> FigureOutput:
    data = df.copy()
    data["Pair Label"] = data["Pair"].map(PAIR_LABELS).fillna(data["Pair"])
    data["Slice Label"] = data["Slice"].str.replace("_", " ", regex=False)
    data["Display"] = data["Slice Label"] + " · " + data["Pair Label"]
    data = data.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(data))
    colors = [PAIR_COLORS.get(str(pair), "#4C78A8") for pair in data["Pair"]]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6), constrained_layout=True)
    ax1, ax2 = axes

    gaps = data["Boundary Gap (s)"].to_numpy(dtype=float)
    bars = ax1.barh(y, gaps, color=colors, height=0.64)
    for yi, v in zip(y, gaps):
        x_text = v + 0.012 if v >= 0 else v - 0.012
        ha = "left" if v >= 0 else "right"
        ax1.text(x_text, yi, f"{v:+.2f}", va="center", ha=ha, fontsize=7)
    ax1.axvline(0.0, color="#707070", linestyle=(0, (3, 2)), linewidth=1.0)
    lim = max(0.5, np.nanmax(np.abs(gaps)) + 0.08)
    ax1.set_xlim(-lim, lim)
    ax1.set_yticks(y, data["Display"])
    ax1.set_xlabel("Signed boundary gap (s)")
    ax1.set_title("(a) Pairwise boundary-gap summaries")
    ax1.grid(axis="x")
    _despine(ax1)

    ranking = data["Ranking Preserved (%)"].to_numpy(dtype=float) / 100.0
    bars = ax2.barh(y, ranking, color="#54A24B", height=0.64)
    for yi, v in zip(y, ranking):
        ax2.text(min(v + 0.01, 0.99), yi, _fmt_pct(v), va="center", fontsize=7)
    ax2.axvline(0.90, color="#808080", linestyle=(0, (3, 2)), linewidth=1.0)
    ax2.set_xlim(0, 1.0)
    ax2.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_yticks(y, data["Display"])
    ax2.set_xlabel("Ranking preserved")
    ax2.set_title("(b) Ranking preservation across pairwise checks")
    ax2.grid(axis="x")
    _despine(ax2)

    source_df = data[["Slice", "Pair", "Boundary Gap (s)", "Ranking Preserved (%)", "Interpretation"]].copy()
    return _source_and_save(fig, "figCHS03_supp_carsim_pairwise_ranking", supp_dir, source_df, source_dir, png_dpi)


def _make_hil_design_figure(df: pd.DataFrame, supp_dir: Path, source_dir: Path, png_dpi: int) -> FigureOutput:
    data = df.copy()
    label_map = {
        "boundary_case_subset_size_total": "Replay count",
        "unique_boundary_groups": "Boundary groups",
        "distinct_sites_count": "Sites",
        "distinct_families_count": "Families",
        "distinct_uncertainty_cells_count": "Uncertainty cells",
        "distinct_core_controllers_count": "Core controllers",
        "boundary_side_count": "Boundary sides",
        "repeats_per_case": "Repeats / case",
        "share_braking_lead": "Share braking_lead",
        "share_lowmu_slow": "Share lowmu_slow",
        "case_overlap_with_carsim_matrix_rate": "Overlap with CarSim matrix",
        "nominal_sample_period_ms": "Nominal sample period",
    }
    data["Label"] = data["Item Key"].map(label_map).fillna(data["Item Label"])

    count_keys = [
        "boundary_case_subset_size_total",
        "unique_boundary_groups",
        "distinct_sites_count",
        "distinct_families_count",
        "distinct_uncertainty_cells_count",
        "distinct_core_controllers_count",
        "boundary_side_count",
        "repeats_per_case",
    ]
    share_keys = [
        "share_braking_lead",
        "share_lowmu_slow",
        "case_overlap_with_carsim_matrix_rate",
    ]
    count_df = data[data["Item Key"].isin(count_keys)].copy()
    count_df["Item Key"] = pd.Categorical(count_df["Item Key"], categories=count_keys, ordered=True)
    count_df = count_df.sort_values("Item Key", ascending=False)
    share_df = data[data["Item Key"].isin(share_keys)].copy()
    share_df["Item Key"] = pd.Categorical(share_df["Item Key"], categories=share_keys, ordered=True)
    share_df = share_df.sort_values("Item Key", ascending=False)
    sample_period = float(data.loc[data["Item Key"] == "nominal_sample_period_ms", "Reported Value"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    ax1, ax2 = axes

    y = np.arange(len(count_df))
    vals = count_df["Reported Value"].to_numpy(dtype=float)
    bars = ax1.barh(y, vals, color="#4C78A8", height=0.62)
    for yi, bar, (_, row) in zip(y, bars, count_df.iterrows()):
        unit = str(row["Unit"])
        suffix = " ms" if unit == "ms" else ""
        ax1.text(bar.get_width() + 0.25, yi, f"{row['Reported Value']:.0f}{suffix}", va="center", fontsize=7)
    ax1.set_yticks(y, count_df["Label"])
    ax1.set_xlabel("Count / value")
    ax1.set_title(f"(a) HIL subset coverage (nominal period = {sample_period:.0f} ms)")
    ax1.grid(axis="x")
    ax1.set_xlim(0, max(36, float(vals.max()) * 1.25))
    _despine(ax1)

    y = np.arange(len(share_df))
    vals = share_df["Reported Value"].to_numpy(dtype=float)
    bars = ax2.barh(y, vals, color="#F58518", height=0.62)
    for yi, bar, (_, row) in zip(y, bars, share_df.iterrows()):
        ax2.text(min(bar.get_width() + 0.015, 0.99), yi, _fmt_pct(float(row["Reported Value"]), 0), va="center", fontsize=7)
    ax2.set_yticks(y, share_df["Label"])
    ax2.set_xlim(0, 1.0)
    ax2.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_xlabel("Share")
    ax2.set_title("(b) HIL subset composition")
    ax2.grid(axis="x")
    _despine(ax2)

    source_df = data[["Item Key", "Label", "Unit", "Reported Value", "Interpretation"]].copy()
    return _source_and_save(fig, "figCHS04_supp_hil_design_subset", supp_dir, source_df, source_dir, png_dpi)


def _write_manifest(output_root: Path, outputs: Iterable[FigureOutput]) -> None:
    rows = []
    for out in outputs:
        rows.append(
            {
                "stem": out.stem,
                "description": out.description,
                "pdf_path": str(out.pdf_path),
                "png_path": str(out.png_path),
                "source_data_path": "" if out.source_data_path is None else str(out.source_data_path),
            }
        )
    pd.DataFrame(rows).to_csv(output_root / "figure_manifest.csv", index=False)


def generate_all_carsim_hil_paper_figures(
    results_root: str | Path,
    output_root: str | Path | None = None,
    png_dpi: int = 600,
) -> list[FigureOutput]:
    _apply_style()
    resolved_root = _resolve_results_root(results_root)
    output_root_path = Path(output_root).expanduser().resolve() if output_root is not None else resolved_root.parent / "carsim_hil_paper_figures"
    main_dir, supp_dir, source_dir = _ensure_output_dirs(output_root_path)

    carsim_main = _read_csv(resolved_root / "01_carsim_main_table.csv")
    carsim_diag = _read_csv(resolved_root / "02_carsim_scatter_and_diagnostic.csv")
    carsim_family = _read_csv(resolved_root / "03_carsim_family_and_cell.csv")
    carsim_pair = _read_csv(resolved_root / "04_carsim_pairwise_ranking.csv")
    hil_main = _read_csv(resolved_root / "05_hil_main_table.csv")
    hil_design = _read_csv(resolved_root / "06_hil_design_and_subset.csv")

    outputs = [
        _make_carsim_summary_figure(carsim_main, main_dir, source_dir, png_dpi),
        _make_hil_summary_figure(hil_main, main_dir, source_dir, png_dpi),
        _make_carsim_diagnostics_figure(carsim_diag, supp_dir, source_dir, png_dpi),
        _make_family_cell_figure(carsim_family, supp_dir, source_dir, png_dpi),
        _make_pairwise_ranking_figure(carsim_pair, supp_dir, source_dir, png_dpi),
        _make_hil_design_figure(hil_design, supp_dir, source_dir, png_dpi),
    ]

    _write_manifest(output_root_path, outputs)
    return outputs
