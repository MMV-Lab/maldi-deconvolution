"""TIFF export and colormap helpers shared by the ``visualize`` and
``registration`` notebooks.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib as mpl
import numpy as np
import tifffile
import xarray as xr
from datashader.transfer_functions import shade
from matplotlib.colors import Colormap, LinearSegmentedColormap

from .imzml_io import pixel_sizes

# Named matplotlib colormaps offered in the visualize notebook's colormap picker.
DEFAULT_COLORMAP_NAMES = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "gray",
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
]

# Custom black -> color linear gradients, layered on top of the named colormaps.
DEFAULT_LINEAR_CONFIGS = {
    "Linear Red": ["black", "red"],
    "Linear Green": ["black", "#00ff00"],
    "Linear Blue": ["black", "blue"],
    "Linear Cyan": ["black", "cyan"],
    "Linear Magenta": ["black", "magenta"],
    "Linear Yellow": ["black", "yellow"],
}


def build_colormap_dict(
    colormap_names: Optional[List[str]] = None,
    linear_configs: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Colormap]:
    """Build a ``name -> matplotlib Colormap`` dict combining named
    colormaps with custom black-to-color linear gradients.

    Pass ``colormap_names=[]`` to get only the linear gradients (as used by
    the registration notebook's overlay colormaps). For the common case of
    "all defaults", use the ready-built `CMAP_DICT` instead of calling this.
    """
    if colormap_names is None:
        colormap_names = DEFAULT_COLORMAP_NAMES
    if linear_configs is None:
        linear_configs = DEFAULT_LINEAR_CONFIGS

    all_names = list(linear_configs.keys()) + list(colormap_names)

    cmap_dict: Dict[str, Colormap] = {}
    for name in all_names:
        if name in linear_configs:
            cmap_dict[name] = LinearSegmentedColormap.from_list(
                name, linear_configs[name]
            )
        else:
            cmap_dict[name] = mpl.colormaps[name]
    return cmap_dict


# Ready-built default colormap dict (named colormaps + linear gradients),
# directly importable without calling build_colormap_dict() yourself.
CMAP_DICT: Dict[str, Colormap] = build_colormap_dict()


def export_xarray_to_tiff(
    data_array: xr.DataArray,
    output_dir: Path,
    base_name: str = "data",
    status_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Export an xarray DataArray to TIFF, one file per m/z value, handling
    both single ion images and multi-iteration stacks.

    The ``resolution`` tag is written in pixels-per-micrometer
    (``1 / pixel_size``, from the Y/X coord spacing), consistent with the
    ``unit: "um"`` ImageJ metadata written alongside it.

    ``base_name`` prefixes every output filename (typically the source
    file's name). ``status_callback``, if given, is called with a short
    progress string before each file is written (used to drive a UI status
    widget).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pixel_size_y, pixel_size_x = pixel_sizes(data_array)
    resolution = (1 / pixel_size_x, 1 / pixel_size_y)

    has_iteration = "ITERATION" in data_array.dims
    mz_vals = data_array.M_Z.values
    total_mz = len(mz_vals)

    for i, mz in enumerate(mz_vals):
        if status_callback is not None:
            status_callback(f"Processing m/z {mz:.4f} ({i + 1}/{total_mz})")

        img = data_array.sel(M_Z=mz).values

        if has_iteration:
            # Select stack: (ITERATION, Y, X)
            axes = "TYX"
            fname = f"{base_name}_mz_{mz:.4f}_deconvolved.tiff"
        else:
            # Select slice: (Y, X)
            axes = "YX"
            fname = f"{base_name}_mz_{mz:.4f}_original.tiff"

        tifffile.imwrite(
            output_dir / fname,
            img.astype(np.float32),
            imagej=True,
            compression="zlib",
            photometric="minisblack",
            resolution=resolution,
            metadata={"axes": axes, "unit": "um"},
        )


def render_shaded_tiff(
    data_2d: xr.DataArray,
    cmap: Colormap,
    span: Tuple[float, float],
    output_path: Path,
    pixel_size_x: float,
    pixel_size_y: float,
) -> None:
    """Render a 2D array to a colorized (datashader-shaded) RGB TIFF."""
    shaded = shade(data_2d, cmap=cmap, span=span, how="linear")
    rgb_array = np.array(shaded.to_pil().convert("RGB"))
    rgb_array = np.flipud(rgb_array)

    tifffile.imwrite(
        output_path,
        rgb_array,
        imagej=True,
        compression="zlib",
        photometric="rgb",
        resolution=(1 / pixel_size_x, 1 / pixel_size_y),
        metadata={"axes": "CYX", "unit": "um"},
    )
