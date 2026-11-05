"""Interactive-plotting (HoloViews/hvplot) helpers shared across notebooks.

Companion to ``src/visualization.py`` (static/export helpers). This module
holds only genuinely reusable, notebook-agnostic interactive-plotting
patterns; one-off exploratory cells stay inline in their notebooks.
"""

import holoviews as hv
import marimo as mo
import pandas as pd
import polars as pl
from bokeh.palettes import Category10, Category20
from typing import Dict, Optional, Sequence


def hv_mpl_svg_render(element: hv.core.Dimensioned, dpi: int = 300) -> mo.Html:
    """Render a HoloViews element to SVG via the matplotlib backend, wrapped
    for marimo display."""
    renderer = hv.plotting.mpl.MPLRenderer.instance(dpi=dpi, fig="svg")
    return mo.Html(renderer.html(element))


def relative_intensity_overlay(
    summary: pl.DataFrame, ylabel: str
) -> hv.NdOverlay:
    """Per-lipid ErrorBars * Curve overlay of relative intensity vs pass.
    Lipids are ordered by abundance (``avg_rel``), so the caller should sort
    ``summary`` on that column first; first-appearance order in ``summary``
    is then used as the ranking."""
    _mzs = summary["m/z"].unique(maintain_order=True).to_list()
    _mz_order = [f"{mz:.2f}" for mz in _mzs]

    overlay = hv.NdOverlay(
        {
            f"{mz:.2f}": hv.ErrorBars(
                summary.filter(pl.col("m/z") == mz),
                kdims="pass_num",
                vdims=["mean_rel", "sem_rel"],
            )
            * hv.Curve(
                summary.filter(pl.col("m/z") == mz),
                kdims="pass_num",
                vdims="mean_rel",
            )
            for mz in _mzs
        },
        kdims=hv.Dimension("m/z", values=_mz_order),
    )
    return overlay.opts(
        hv.opts.Curve(),
        hv.opts.NdOverlay(
            xlabel="Pass", ylabel=ylabel, legend_position="right"
        ),
    )


def decay_curve_group_overlay(
    df: pl.DataFrame,
    group_col: str,
    x_col: str,
    y_col: str,
    yerr_col: str,
    *,
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    title: str = "",
    xlabel: str = "Pass",
    ylabel: str = "Normalised Intensity",
) -> hv.NdOverlay:
    """Build a per-group ErrorBars * Curve * Scatter decay-curve overlay.

    One overlay entry is produced per distinct value of ``group_col`` (e.g.
    laser-energy setting or tissue type), each styled with a distinct
    color/marker (auto-cycled from a predefined palette if not given).
    """
    default_markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    groups = list(dict.fromkeys(df[group_col].to_list()))

    n = len(groups)
    if n <= 3:
        palette = Category10[3][:n]
    elif n <= 10:
        palette = Category10[n]
    elif n <= 20:
        palette = Category20[n]
    else:
        base = Category20[20]
        palette = [base[i % len(base)] for i in range(n)]

    colors = colors or {g: palette[i] for i, g in enumerate(groups)}
    markers = markers or {
        g: default_markers[i % len(default_markers)]
        for i, g in enumerate(groups)
    }

    overlays = {}
    for g in groups:
        sub = df.filter(pl.col(group_col) == g)
        overlays[g] = (
            hv.ErrorBars(sub, kdims=[x_col], vdims=[y_col, yerr_col]).opts(
                color=colors[g], alpha=0.4
            )
            * hv.Curve(
                sub, kdims=[x_col], vdims=[y_col], label=str(g)
            ).opts(color=colors[g])
            * hv.Scatter(sub, kdims=[x_col], vdims=[y_col]).opts(
                color=colors[g], marker=markers[g], s=50
            )
        )

    return hv.NdOverlay(overlays, kdims=[hv.Dimension(group_col)]).opts(
        hv.opts.Curve(xlabel=xlabel, ylabel=ylabel),
        hv.opts.NdOverlay(title=title),
    )


def image_comparison_layout(
    plots: Sequence[hv.Element],
    cols: int,
    *,
    opts_group: str = "Image",
    title: str = "",
    shared_axes: Optional[bool] = None,
    **group_opts,
) -> hv.Layout:
    """Arrange plots into a shared side-by-side comparison grid:
    ``hv.Layout(plots).cols(cols)`` with a single ``hv.opts.<opts_group>``
    applied to every element plus a shared ``hv.opts.Layout(title=...)``.

    ``opts_group`` selects which element type the style options apply to
    (``"Image"`` for plain image panels, ``"Overlay"`` for image*label
    overlays). ``group_opts`` are passed straight through, e.g.
    ``invert_yaxis=True, frame_height=200, frame_width=200``.
    """
    element_opts = getattr(hv.opts, opts_group)(**group_opts)
    layout_kwargs = {"title": title}
    if shared_axes is not None:
        layout_kwargs["shared_axes"] = shared_axes
    return (
        hv.Layout(list(plots))
        .cols(cols)
        .opts(element_opts, hv.opts.Layout(**layout_kwargs))
    )
