"""Interactive-plotting (HoloViews/hvplot) helpers shared across notebooks.

Companion to ``src/visualization.py`` (static/export helpers). This module
holds only genuinely reusable, notebook-agnostic interactive-plotting
patterns; one-off exploratory cells stay inline in their notebooks.
"""

from typing import Dict, Optional, Sequence

import holoviews as hv
import pandas as pd


def set_hv_backend(backend: str, fig_format: Optional[str] = "svg") -> None:
    """Switch HoloViews' active rendering backend, optionally pinning the
    static export format (``fig=`` used by ``hv.save``/SVG output).

    Replaces the repeated ``hv.extension(backend)`` / ``hv.extension(backend);
    hv.output(backend=backend, fig=fig_format)`` pairs scattered through the
    notebooks. Pass ``fig_format=None`` to only switch backends (e.g. quick
    bokeh interactivity for one cell) without touching output format.
    """
    hv.extension(backend)
    if fig_format is not None:
        hv.output(backend=backend, fig=fig_format)


def register_hv_backends(*backends: str, fig_format: str = "svg") -> None:
    """Register multiple hvplot/HoloViews backends up front (module-setup
    time) and set the default static export format for hvplot's own output.

    Used once, in a notebook's setup block, when a notebook needs to
    fluidly switch between backends later via ``set_hv_backend``.
    """
    import hvplot

    hvplot.extension(*backends)
    hvplot.output(fig=fig_format)


def decay_curve_group_overlay(
    df: pd.DataFrame,
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
    color/marker (auto-cycled from a default palette if not given).
    """
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    default_markers = ["o", "s", "^", "D", "v"]
    groups = list(df[group_col].unique())
    colors = colors or {
        g: default_colors[i % len(default_colors)] for i, g in enumerate(groups)
    }
    markers = markers or {
        g: default_markers[i % len(default_markers)] for i, g in enumerate(groups)
    }

    overlays = {}
    for g in groups:
        sub = df[df[group_col] == g]
        overlays[g] = (
            hv.ErrorBars(sub, kdims=[x_col], vdims=[y_col, yerr_col]).opts(
                color=colors[g], alpha=0.4
            )
            * hv.Curve(sub, kdims=[x_col], vdims=[y_col], label=str(g)).opts(
                color=colors[g]
            )
            * hv.Scatter(sub, kdims=[x_col], vdims=[y_col]).opts(
                color=colors[g], marker=markers[g], s=50
            )
        )
    return hv.NdOverlay(overlays).opts(
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
