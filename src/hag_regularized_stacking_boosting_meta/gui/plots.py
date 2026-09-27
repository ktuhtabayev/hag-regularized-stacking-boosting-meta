"""
Matplotlib drawings for the GUI (no Qt here, so they can be tested headless).

  - HAG criterion θ/γ by Step-3/4 iteration, with the δ stop threshold
  - Margin strip: one latent feature r_j, both classes, boundaries and margin band
  - Margin width by latent feature r_1..r_p (how regularization widens the margin)
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from hag_regularized_stacking_boosting_meta.gui.theme import PLOT_COLORS, THEME
from hag_regularized_stacking_boosting_meta.services.runner import PipelineResult


LINE_WIDTH = 2.0
MARKER_SIZE = 7.0          # points (~9 px): readable markers with a surface ring
RING_WIDTH = 1.5
_GOLDEN = 0.6180339887498949


def style_axis(axis: Axes) -> None:
    """Recessive chrome: white plot, solid hairline grid, no top/right spines."""
    axis.set_facecolor(THEME["axes_bg"])
    axis.grid(True, color=THEME["plot_grid_line"], linewidth=0.8, linestyle="-")
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(THEME["border"])
    axis.tick_params(colors=PLOT_COLORS["ink_muted"], labelsize=9)
    axis.title.set_color(PLOT_COLORS["ink"])
    axis.xaxis.label.set_color(PLOT_COLORS["ink"])
    axis.yaxis.label.set_color(PLOT_COLORS["ink"])


def _message(axis: Axes, text: str) -> None:
    axis.text(0.5, 0.5, text, ha="center", va="center", transform=axis.transAxes,
              color=PLOT_COLORS["ink_muted"])
    axis.set_xticks([])
    axis.set_yticks([])


def draw_criterion_history(axis: Axes, result: PipelineResult) -> None:
    style_axis(axis)
    hag = result.hag
    crit = [float(c) for c in hag.crit_history]
    if not crit:
        _message(axis, "HAG made no Step-3 selection (single feature).")
        return

    steps = list(range(1, len(crit) + 1))
    axis.plot(
        steps,
        crit,
        color=PLOT_COLORS["series"],
        linewidth=LINE_WIDTH,
        solid_capstyle="round",
        solid_joinstyle="round",
        marker="o",
        markersize=MARKER_SIZE,
        markeredgecolor=THEME["surface"],
        markeredgewidth=RING_WIDTH,
    )

    delta = float(result.config.hag.delta)
    axis.axhline(delta, color=PLOT_COLORS["ink_muted"], linewidth=1.0)
    axis.text(
        steps[-1],
        delta,
        f"  δ = {delta:g} (stop when θ/γ ≤ δ)",
        ha="right",
        va="bottom",
        fontsize=8,
        color=PLOT_COLORS["ink_muted"],
    )

    best = int(np.argmin(crit))
    axis.annotate(
        f"min θ/γ = {crit[best]:.4f}",
        (steps[best], crit[best]),
        xytext=(0, -16),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color=PLOT_COLORS["ink"],
    )

    axis.set_xticks(steps)
    axis.set_xticklabels([f"{t}\nx{q}" for t, q in zip(steps, hag.tuplam[1:])])
    low = min(min(crit), delta)
    high = max(max(crit), delta)
    pad = max((high - low) * 0.15, 0.01)
    axis.set_ylim(low - pad, high + pad)
    axis.set_title(f"HAG criterion θ/γ by step (organizer u = x{hag.organizer})")
    axis.set_xlabel("Step t (selected feature q)")
    axis.set_ylabel("θ/γ")


def _jitter(count: int) -> np.ndarray:
    """Deterministic vertical spread in [-0.18, 0.18] so equal values stay visible."""
    return ((np.arange(count) * _GOLDEN) % 1.0 - 0.5) * 0.36


def draw_margin_strip(axis: Axes, result: PipelineResult, latent_index: int) -> None:
    style_axis(axis)
    if not result.margins:
        _message(axis, "No latent features (p = 0): margin analysis is not available.")
        return

    margin = result.margins[latent_index]
    d = result.hag.dij[:, latent_index]
    y = result.dataset.y
    k1, k2 = result.k1_label, result.k2_label
    left, right, mid = margin.left_boundary, margin.right_boundary, margin.midpoint

    overlap = margin.width < 0
    axis.axvspan(
        min(left, right),
        max(left, right),
        color=PLOT_COLORS["overlap"] if overlap else PLOT_COLORS["margin_band"],
        alpha=0.12 if overlap else 1.0,
        linewidth=0,
    )
    axis.axvline(right, color=PLOT_COLORS["class_k1"], linewidth=1.5)
    axis.axvline(left, color=PLOT_COLORS["class_k2"], linewidth=1.5)
    axis.axvline(mid, color=PLOT_COLORS["ink"], linewidth=1.0)

    top = blended_transform_factory(axis.transData, axis.transAxes)
    for x, text, align in (
        (left, "max K2", "right" if left <= right else "left"),
        (mid, "mid", "center"),
        (right, "min K1", "left" if left <= right else "right"),
    ):
        axis.text(x, 1.01, text, transform=top, ha=align, va="bottom", fontsize=8,
                  color=PLOT_COLORS["ink_muted"])

    rows = {k1: 1.0, k2: 0.0}
    colors = {k1: PLOT_COLORS["class_k1"], k2: PLOT_COLORS["class_k2"]}
    spread = _jitter(len(y))
    wrong = margin.yhat != y
    for label in (k1, k2):
        in_class = y == label
        for mask, hollow in ((in_class & ~wrong, False), (in_class & wrong, True)):
            if not mask.any():
                continue
            axis.scatter(
                d[mask],
                rows[label] + spread[mask],
                s=MARKER_SIZE ** 2 * (1.4 if hollow else 1.0),
                facecolors="none" if hollow else colors[label],
                edgecolors=colors[label] if hollow else THEME["surface"],
                linewidths=2.0 if hollow else RING_WIDTH,
                zorder=3,
            )

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=MARKER_SIZE,
               markerfacecolor=PLOT_COLORS["class_k1"], markeredgecolor=THEME["surface"],
               label=f"K1 (class {k1})"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=MARKER_SIZE,
               markerfacecolor=PLOT_COLORS["class_k2"], markeredgecolor=THEME["surface"],
               label=f"K2 (class {k2})"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=MARKER_SIZE,
               markerfacecolor="none", markeredgecolor=PLOT_COLORS["ink_muted"],
               markeredgewidth=2.0, label="misclassified (ŷ ≠ class)"),
    ]
    axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
                frameon=False, fontsize=8)

    span = max(float(d.max() - d.min()), 1e-9)
    axis.set_xlim(float(d.min()) - 0.05 * span, float(d.max()) + 0.05 * span)
    axis.set_ylim(-0.6, 1.6)
    axis.set_yticks([0.0, 1.0])
    axis.set_yticklabels([f"K2 ({k2})", f"K1 ({k1})"])
    axis.grid(False, axis="y")
    state = "overlap" if overlap else "margin"
    axis.set_title(
        f"r{latent_index + 1}: {state} width = {margin.width:.4f}, "
        f"misclassified {int(wrong.sum())} of {len(y)}",
        pad=16,
    )
    axis.set_xlabel(f"Latent value d(r{latent_index + 1})")


def draw_margin_widths(axis: Axes, result: PipelineResult, current_index: int) -> None:
    style_axis(axis)
    if not result.margins:
        _message(axis, "No latent features (p = 0).")
        return

    widths = [float(m.width) for m in result.margins]
    steps = list(range(1, len(widths) + 1))
    axis.axhline(0.0, color=PLOT_COLORS["ink_muted"], linewidth=1.0)
    axis.plot(
        steps,
        widths,
        color=PLOT_COLORS["series"],
        linewidth=LINE_WIDTH,
        solid_capstyle="round",
        solid_joinstyle="round",
        marker="o",
        markersize=MARKER_SIZE,
        markeredgecolor=THEME["surface"],
        markeredgewidth=RING_WIDTH,
    )
    current = int(current_index)
    axis.scatter([steps[current]], [widths[current]], s=(MARKER_SIZE * 2.2) ** 2,
                 facecolors="none", edgecolors=PLOT_COLORS["ink"], linewidths=1.5, zorder=4)
    axis.annotate(
        f"{widths[current]:.4f}",
        (steps[current], widths[current]),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color=PLOT_COLORS["ink"],
    )
    axis.set_xticks(steps)
    axis.set_xticklabels([f"r{s}" for s in steps])
    low, high = min(min(widths), 0.0), max(max(widths), 0.0)
    pad = max((high - low) * 0.18, 0.05)
    axis.set_ylim(low - pad, high + pad)
    axis.set_title("Margin width by latent feature (< 0 = classes overlap)")
    axis.set_ylabel("Width")
