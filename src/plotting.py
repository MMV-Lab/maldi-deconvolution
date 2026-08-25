"""Interactive-plotting (HoloViews/hvplot) helpers shared across notebooks.

Companion to ``src/visualization.py`` (static/export helpers). This module
holds only genuinely reusable, notebook-agnostic interactive-plotting
patterns; one-off exploratory cells stay inline in their notebooks.
"""

import holoviews as hv
import pandas as pd
from typing import Dict, Optional, Sequence


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
