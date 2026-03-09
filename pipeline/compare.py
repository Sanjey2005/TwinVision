"""
Winner determination, chart generation, and comparison report for TwinVision.

Consumes the metrics DataFrame from evaluate.py to:
  - Compute a weighted winner using METRIC_WEIGHTS from config
  - Generate dark-mode matplotlib bar and radar charts
  - Build an image grid comparing first images per prompt
  - Assemble the ComparisonPayload dict returned by the API
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend; must be set before pyplot import
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from pipeline.config import (
    CHART_C_BG,
    CHART_C_FLUX,
    CHART_C_GRID,
    CHART_C_MUTED,
    CHART_C_PANEL,
    CHART_C_SD35,
    CHART_C_TEXT,
    METRIC_WEIGHTS,
    OUTPUT_BASE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Metrics where a lower raw score is better.
_LOWER_IS_BETTER: frozenset[str] = frozenset({"brisque", "niqe", "lpips"})

# Human-readable display names.
_METRIC_DISPLAY: dict[str, str] = {
    "clip_score": "CLIP Score",
    "brisque":    "BRISQUE",
    "niqe":       "NIQE",
    "ssim":       "SSIM",
    "lpips":      "LPIPS",
}
_MODEL_FULL: dict[str, str] = {
    "flux": "FLUX.1 [schnell]",
    "sd35": "SD3.5 Large",
}
_MODEL_SHORT: dict[str, str] = {
    "flux": "FLUX",
    "sd35": "SD3.5",
}




# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pivot_mean(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return mean average_score per (model, metric) across all prompts.

    Args:
        metrics_df: DataFrame produced by evaluate.run_all_metrics().

    Returns:
        DataFrame with index ['flux', 'sd35'] and metric names as columns.
    """
    return (
        metrics_df.groupby(["model", "metric"])["average_score"]
        .mean()
        .unstack("metric")
    )


def _safe_score(pivot: pd.DataFrame, model: str, metric: str) -> float:
    """Extract a score from the pivot table, returning 0.0 if absent.

    Args:
        pivot: Result of _pivot_mean().
        model: Model key ('flux' or 'sd35').
        metric: Metric name.

    Returns:
        Float score, or 0.0 if the (model, metric) combination is missing.
    """
    try:
        return float(pivot.loc[model, metric])
    except (KeyError, TypeError):
        return 0.0


def _normalise_pair(flux_raw: float, sd35_raw: float, metric: str) -> tuple[float, float]:
    """Min-max normalise two model scores to [0, 1] where 1 = better.

    For lower-is-better metrics the mapping is inverted so that the model
    with the lower raw score receives the higher normalised score.
    When both scores are equal both receive 0.5 (tie).

    Args:
        flux_raw: Raw average score for FLUX.1.
        sd35_raw: Raw average score for SD3.5.
        metric: Metric name; checked against _LOWER_IS_BETTER.

    Returns:
        Tuple (flux_norm, sd35_norm), each in [0, 1].
    """
    lo, hi = min(flux_raw, sd35_raw), max(flux_raw, sd35_raw)
    if hi == lo:
        return 0.5, 0.5
    span = hi - lo
    if metric in _LOWER_IS_BETTER:
        return (hi - flux_raw) / span, (hi - sd35_raw) / span
    return (flux_raw - lo) / span, (sd35_raw - lo) / span


def _winner_key(flux_raw: float, sd35_raw: float, metric: str) -> str:
    """Return which model wins a metric ('flux', 'sd35', or 'tie').

    Args:
        flux_raw: Raw average score for FLUX.1.
        sd35_raw: Raw average score for SD3.5.
        metric: Metric name; checked against _LOWER_IS_BETTER.

    Returns:
        'flux', 'sd35', or 'tie'.
    """
    if metric in _LOWER_IS_BETTER:
        if flux_raw < sd35_raw:
            return "flux"
        if sd35_raw < flux_raw:
            return "sd35"
    else:
        if flux_raw > sd35_raw:
            return "flux"
        if sd35_raw > flux_raw:
            return "sd35"
    return "tie"


def _diff_pct(flux_raw: float, sd35_raw: float) -> float:
    """Compute relative percentage difference between two scores.

    Formula: abs(a - b) / max(abs(a), abs(b)) * 100

    Args:
        flux_raw: Score for FLUX.1.
        sd35_raw: Score for SD3.5.

    Returns:
        Percentage difference rounded to one decimal place.
    """
    denom = max(abs(flux_raw), abs(sd35_raw))
    if denom == 0.0:
        return 0.0
    return round(abs(flux_raw - sd35_raw) / denom * 100, 1)


def _apply_dark_axes(ax: plt.Axes) -> None:
    """Apply consistent dark-mode styling to a Cartesian Axes object.

    Args:
        ax: The Axes to style in-place.
    """
    ax.set_facecolor(CHART_C_PANEL)
    ax.tick_params(colors=CHART_C_TEXT, labelsize=9)
    ax.xaxis.label.set_color(CHART_C_TEXT)
    ax.yaxis.label.set_color(CHART_C_TEXT)
    ax.title.set_color(CHART_C_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(CHART_C_GRID)
    ax.grid(color=CHART_C_GRID, linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)


# ---------------------------------------------------------------------------
# Public: winner computation
# ---------------------------------------------------------------------------

def compute_weighted_winner(metrics_df: pd.DataFrame) -> dict:
    """Compute the overall and per-metric winners using METRIC_WEIGHTS.

    Each metric's raw scores are min-max normalised to [0, 1] (inverted for
    lower-is-better metrics), then combined via a weighted sum.

    Args:
        metrics_df: DataFrame produced by evaluate.run_all_metrics().

    Returns:
        Dict containing:
            overall_winner (str): Display name of the winning model or 'Tie'.
            overall_score (dict): Weighted scores {'flux': float, 'sd35': float}.
            metric_winners (dict): Per-metric dicts with keys
                winner (str), flux (float), sd35 (float), diff_pct (float).
            normalized_scores (dict): Per-model per-metric normalised scores
                in [0, 1]; used by generate_radar_chart().
    """
    pivot = _pivot_mean(metrics_df)
    metric_names = list(METRIC_WEIGHTS.keys())

    weighted_flux = 0.0
    weighted_sd35 = 0.0
    metric_winners: dict[str, dict] = {}
    norm_scores: dict[str, dict[str, float]] = {"flux": {}, "sd35": {}}

    for metric in metric_names:
        weight = METRIC_WEIGHTS[metric]
        flux_raw = _safe_score(pivot, "flux", metric)
        sd35_raw = _safe_score(pivot, "sd35", metric)

        fn, sn = _normalise_pair(flux_raw, sd35_raw, metric)
        norm_scores["flux"][metric] = fn
        norm_scores["sd35"][metric] = sn
        weighted_flux += weight * fn
        weighted_sd35 += weight * sn

        wk = _winner_key(flux_raw, sd35_raw, metric)
        metric_winners[metric] = {
            "winner":   _MODEL_SHORT.get(wk, "Tie"),
            "flux":     round(flux_raw, 4),
            "sd35":     round(sd35_raw, 4),
            "diff_pct": _diff_pct(flux_raw, sd35_raw),
        }

    weighted_flux = round(weighted_flux, 4)
    weighted_sd35 = round(weighted_sd35, 4)

    if weighted_flux > weighted_sd35:
        overall_winner = _MODEL_FULL["flux"]
    elif weighted_sd35 > weighted_flux:
        overall_winner = _MODEL_FULL["sd35"]
    else:
        overall_winner = "Tie"

    logger.info(
        "Weighted scores — FLUX: %.4f  SD3.5: %.4f  →  Winner: %s",
        weighted_flux, weighted_sd35, overall_winner,
    )

    return {
        "overall_winner":    overall_winner,
        "overall_score":     {"flux": weighted_flux, "sd35": weighted_sd35},
        "metric_winners":    metric_winners,
        "normalized_scores": norm_scores,
    }


# ---------------------------------------------------------------------------
# Public: chart generation
# ---------------------------------------------------------------------------

def generate_bar_chart(
    metrics_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """Generate a grouped bar chart of raw metric averages for both models.

    Displays the mean average_score per model per metric across all prompts.
    Each metric bar is annotated with its raw value and a direction hint
    (↑ better / ↓ better).

    Args:
        metrics_df: DataFrame from evaluate.run_all_metrics().
        output_dir: Directory to write bar_chart.png. Defaults to
            OUTPUT_BASE/results/.

    Returns:
        Path to the saved bar_chart.png.
    """
    save_dir = output_dir if output_dir is not None else OUTPUT_BASE / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "bar_chart.png"

    pivot = _pivot_mean(metrics_df)
    metric_names  = list(METRIC_WEIGHTS.keys())
    display_names = [_METRIC_DISPLAY[m] for m in metric_names]
    n = len(metric_names)

    flux_vals = [_safe_score(pivot, "flux", m) for m in metric_names]
    sd35_vals = [_safe_score(pivot, "sd35", m) for m in metric_names]

    x     = np.arange(n)
    bar_w = 0.35

    fig = plt.figure(figsize=(13, 6), facecolor=CHART_C_BG)
    ax  = fig.add_subplot(111)
    _apply_dark_axes(ax)

    bars_f = ax.bar(x - bar_w / 2, flux_vals, bar_w,
                    label=_MODEL_FULL["flux"], color=CHART_C_FLUX, alpha=0.85, zorder=3)
    bars_s = ax.bar(x + bar_w / 2, sd35_vals, bar_w,
                    label=_MODEL_FULL["sd35"], color=CHART_C_SD35, alpha=0.85, zorder=3)

    # Value labels above each bar.
    for bar, color in [(b, CHART_C_FLUX) for b in bars_f] + [(b, CHART_C_SD35) for b in bars_s]:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom",
            color=color, fontsize=8, fontweight="bold",
        )

    # Direction hints below x-tick labels.
    y_hint = ax.get_ylim()[0]
    for i, metric in enumerate(metric_names):
        hint = "↓ better" if metric in _LOWER_IS_BETTER else "↑ better"
        ax.text(i, y_hint, hint, ha="center", va="top",
                color=CHART_C_MUTED, fontsize=7, fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.set_ylabel("Average Score", fontsize=11)
    ax.set_title(
        "TwinVision — Model Metric Comparison",
        color=CHART_C_TEXT, fontsize=14, fontweight="bold", pad=15,
    )
    ax.legend(facecolor=CHART_C_PANEL, edgecolor=CHART_C_GRID, labelcolor=CHART_C_TEXT, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=CHART_C_BG)
    plt.close(fig)
    logger.info("Bar chart saved: %s", out_path)
    return out_path


def generate_radar_chart(
    summary: dict,
    output_dir: Path | None = None,
) -> Path:
    """Generate a radar / spider chart from normalised per-metric scores.

    Both models are plotted as filled polygons on the same polar axes.
    Normalised scores ensure all five metrics share the same [0, 1] scale
    where 1 always represents the better direction.

    Args:
        summary: Return value of compute_weighted_winner(); must contain
            'normalized_scores' with per-model per-metric values.
        output_dir: Directory to write radar_chart.png. Defaults to
            OUTPUT_BASE/results/.

    Returns:
        Path to the saved radar_chart.png.
    """
    save_dir = output_dir if output_dir is not None else OUTPUT_BASE / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "radar_chart.png"

    metric_names  = list(METRIC_WEIGHTS.keys())
    display_names = [_METRIC_DISPLAY[m] for m in metric_names]
    n = len(metric_names)

    norm = summary.get("normalized_scores", {})
    flux_vals = [norm.get("flux", {}).get(m, 0.0) for m in metric_names]
    sd35_vals = [norm.get("sd35", {}).get(m, 0.0) for m in metric_names]

    # Polar requires the polygon to be closed by repeating the first point.
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles    += angles[:1]
    flux_vals += flux_vals[:1]
    sd35_vals += sd35_vals[:1]

    fig = plt.figure(figsize=(8, 8), facecolor=CHART_C_BG)
    ax  = fig.add_subplot(111, projection="polar")
    ax.set_facecolor(CHART_C_PANEL)

    # Orientation: first spoke at top, clockwise.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Radial axis.
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                        color=CHART_C_MUTED, fontsize=7)
    ax.set_rlabel_position(30)
    ax.grid(color=CHART_C_GRID, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.spines["polar"].set_color(CHART_C_GRID)

    # Spoke labels.
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_names, color=CHART_C_TEXT, fontsize=11)

    # FLUX polygon.
    ax.plot(angles, flux_vals, color=CHART_C_FLUX, linewidth=2,
            label=_MODEL_FULL["flux"])
    ax.fill(angles, flux_vals, color=CHART_C_FLUX, alpha=0.20)

    # SD3.5 polygon.
    ax.plot(angles, sd35_vals, color=CHART_C_SD35, linewidth=2,
            label=_MODEL_FULL["sd35"])
    ax.fill(angles, sd35_vals, color=CHART_C_SD35, alpha=0.20)

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.30, 1.15),
        facecolor=CHART_C_PANEL, edgecolor=CHART_C_GRID,
        labelcolor=CHART_C_TEXT, fontsize=10,
    )
    ax.set_title(
        "Normalised Performance Radar\n(1.0 = better for all metrics)",
        color=CHART_C_TEXT, fontsize=12, fontweight="bold", pad=20,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=CHART_C_BG)
    plt.close(fig)
    logger.info("Radar chart saved: %s", out_path)
    return out_path


def generate_image_grid(
    flux_images: dict[int, list[Path]],
    sd35_images: dict[int, list[Path]],
    job_id: str = "default",
    output_dir: Path | None = None,
) -> Path:
    """Create a 2×N image grid comparing first outputs per prompt.

    Layout: row 0 = FLUX.1, row 1 = SD3.5; one column per prompt index.
    Only the first image from each model/prompt is used.
    Missing images are replaced by a placeholder cell.

    Args:
        flux_images: Dict mapping prompt_index → list of FLUX image Paths.
        sd35_images: Dict mapping prompt_index → list of SD3.5 image Paths.
        job_id: Unique job identifier; used to scope the default output path.
        output_dir: Directory to write image_grid.png. Defaults to
            OUTPUT_BASE/job_id/results/.

    Returns:
        Path to the saved image_grid.png.
    """
    save_dir = (output_dir if output_dir is not None
                else OUTPUT_BASE / job_id / "results")
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "image_grid.png"

    all_indices = sorted(set(flux_images) | set(sd35_images))
    n_cols = len(all_indices)

    if n_cols == 0:
        logger.warning("generate_image_grid: no images provided — skipping.")
        return out_path

    thumb_px   = 256    # thumbnail side length in pixels
    fig_w      = max(n_cols * thumb_px / 96, 4.0)
    fig_h      = 2 * thumb_px / 96 + 0.6   # 2 image rows + label row height

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=CHART_C_BG)
    # 3 rows: header + FLUX + SD3.5
    gs = gridspec.GridSpec(
        3, n_cols, figure=fig,
        height_ratios=[0.12, 1, 1],
        hspace=0.04, wspace=0.03,
    )

    # Column headers.
    for col, idx in enumerate(all_indices):
        ax_h = fig.add_subplot(gs[0, col])
        ax_h.axis("off")
        ax_h.set_facecolor(CHART_C_BG)
        ax_h.text(0.5, 0.5, f"Prompt {idx}",
                  ha="center", va="center",
                  color=CHART_C_MUTED, fontsize=8,
                  transform=ax_h.transAxes)

    def _render_cell(
        row: int,
        col: int,
        img_path: Path | None,
        row_label: str,
        accent: str,
    ) -> None:
        """Render a single thumbnail cell in the grid.

        Args:
            row: Grid row index (1 = FLUX, 2 = SD3.5).
            col: Grid column index.
            img_path: Path to the image, or None if unavailable.
            row_label: Label text shown on the leftmost cell of the row.
            accent: Hex colour for border and label.
        """
        ax = fig.add_subplot(gs[row, col])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(CHART_C_PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(accent)
            spine.set_linewidth(1.5)

        if col == 0:
            ax.set_ylabel(row_label, color=accent, fontsize=9,
                          fontweight="bold", rotation=90, labelpad=4)

        if img_path is not None and img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((thumb_px, thumb_px), Image.LANCZOS)
                ax.imshow(img)
                return
            except (OSError, ValueError) as exc:
                logger.warning("Cannot load image %s: %s", img_path, exc)

        ax.text(0.5, 0.5, "Missing", ha="center", va="center",
                color=CHART_C_MUTED, fontsize=8, transform=ax.transAxes)

    model_rows: list[tuple[int, dict[int, list[Path]], str, str]] = [
        (1, flux_images, "FLUX.1",  CHART_C_FLUX),
        (2, sd35_images, "SD3.5",   CHART_C_SD35),
    ]
    for row_idx, img_dict, label, color in model_rows:
        for col, prompt_idx in enumerate(all_indices):
            paths = img_dict.get(prompt_idx, [])
            _render_cell(row_idx, col, paths[0] if paths else None, label, color)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=CHART_C_BG)
    plt.close(fig)
    logger.info("Image grid saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Public: full report assembly
# ---------------------------------------------------------------------------

def build_comparison_report(
    metrics_df: pd.DataFrame,
    job_id: str,
    flux_images: dict[int, list[Path]] | None = None,
    sd35_images: dict[int, list[Path]] | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Assemble the full ComparisonPayload dict for the API layer.

    Calls compute_weighted_winner, generate_bar_chart, generate_radar_chart,
    and (when both image dicts are supplied) generate_image_grid.

    Args:
        metrics_df: DataFrame from evaluate.run_all_metrics().
        job_id: Unique job identifier; used to scope the output directory.
        flux_images: Optional dict of prompt_index → FLUX image Paths.
            Required to include image_grid in the charts dict.
        sd35_images: Optional dict of prompt_index → SD3.5 image Paths.
        output_dir: Directory for chart PNGs. Defaults to
            OUTPUT_BASE/job_id/results/.

    Returns:
        Dict matching the ComparisonPayload shape:
            overall_winner (str)
            overall_score  (dict[str, float])
            metric_winners (dict[str, dict])
            charts         (dict[str, str])  — filesystem paths to PNGs
    """
    save_dir = (output_dir if output_dir is not None
                else OUTPUT_BASE / job_id / "results")
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Computing weighted winner...")
    summary = compute_weighted_winner(metrics_df)

    logger.info("Generating bar chart...")
    bar_path = generate_bar_chart(metrics_df, output_dir=save_dir)

    logger.info("Generating radar chart...")
    radar_path = generate_radar_chart(summary, output_dir=save_dir)

    charts: dict[str, str] = {
        "bar_chart":   str(bar_path),
        "radar_chart": str(radar_path),
    }

    if flux_images and sd35_images:
        logger.info("Generating image grid...")
        grid_path = generate_image_grid(
            flux_images=flux_images,
            sd35_images=sd35_images,
            job_id=job_id,
            output_dir=save_dir,
        )
        charts["image_grid"] = str(grid_path)

    report: dict[str, Any] = {
        "overall_winner": summary["overall_winner"],
        "overall_score":  summary["overall_score"],
        "metric_winners": summary["metric_winners"],
        "charts":         charts,
    }

    logger.info(
        "Report built — Winner: %s  (FLUX=%.4f, SD3.5=%.4f).",
        report["overall_winner"],
        report["overall_score"]["flux"],
        report["overall_score"]["sd35"],
    )
    return report
