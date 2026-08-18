"""MSI I/O through BioIO: imzML files via bioio-imzml, OME-Zarr stores via
bioio-ome-zarr, everything read through ``BioImage`` so pixel sizes and m/z
values always come from the standard BioImage metadata
(``physical_pixel_sizes``, ``channel_names``) instead of anything custom.

``bioimage_to_dataarray`` converts a ``BioImage`` into this project's MSI
DataArray convention: dims ``(M_Z, Y, X)`` with float m/z coords parsed
from the channel names, Y/X coords in micrometers derived from the pixel
size, plus a leading ``ITERATION`` dim mapped from the OME time axis when
present. ``save_dataarray`` writes that convention back to an OME-Zarr
store (the only on-disk format this project uses); the only metadata
persisted is standard OME-NGFF: m/z as the channel labels, pixel size as
the spatial axis scales.

Shared by the ``extract_data``, ``extract_decay_profile``, ``deconvolve``,
``registration``, ``metrics``, ``figures``, and ``visualize`` notebooks.
"""

from collections.abc import Sequence
from pathlib import Path

import bioio_imzml
import numpy as np
import polars as pl
import xarray as xr
from bioio import BioImage
from bioio_ome_zarr.writers import Channel, OMEZarrWriter

# The only on-disk format save_dataarray/load_dataarray support, for
# notebooks to pass straight to `mo.ui.file_browser(filetypes=...)`.
DATAARRAY_EXTENSIONS = [".ome.zarr"]

MSI_DIMS = ("M_Z", "Y", "X")


def read_imzml_to_xarray(
    filepath: str | Path,
    m_z_values: list[float],
    tolerance: float = 0.005,
) -> xr.DataArray:
    """Extract ion images from an imzML file via bioio-imzml.

    Parameters
    ----------
    filepath:
        Path to the .imzml file (a sibling .ibd must exist).
    m_z_values:
        Target m/z values to extract, one channel each.
    tolerance:
        Absolute matching window in Da around each target
        (bioio-imzml's ``mz_tolerance_absolute``).

    Returns
    -------
    xr.DataArray
        Dims ``(M_Z, Y, X)``, read eagerly (see `bioimage_to_dataarray`).
    """
    img = BioImage(
        filepath,
        reader=bioio_imzml.Reader,
        mz=m_z_values,
        mz_tolerance_absolute=tolerance,
    )
    return bioimage_to_dataarray(img, lazy=False)


def bioimage_to_dataarray(img: BioImage, lazy: bool = True) -> xr.DataArray:
    """Convert a ``BioImage`` into this project's MSI DataArray convention.

    The OME channel axis becomes ``M_Z``, with float coords parsed from
    ``img.channel_names`` (bioio-imzml formats them as ``"<m/z>±<tol>"``,
    `save_dataarray` writes plain floats; a non-numeric label falls back to
    its channel index). The OME time axis, if non-singleton, becomes
    ``ITERATION`` with plain 0..T-1 coords. Y/X coords are micrometers,
    ``arange(n) * pixel_size`` from ``img.physical_pixel_sizes``. The
    singleton Z axis is dropped.

    lazy:
        True -- dask-backed (``img.xarray_dask_data``); required before
        `.chunk()`-ing for dask-parallel processing.
        False -- read eagerly (``img.xarray_data``), preferable when the
        array will be sliced/indexed repeatedly afterward.
    """
    arr = img.xarray_dask_data if lazy else img.xarray_data

    if arr.sizes["Z"] != 1:
        raise ValueError(
            f"Expected 2D MSI data (Z=1), got Z={arr.sizes['Z']} in {img.reader}"
        )
    arr = arr.isel(Z=0, drop=True)

    if arr.sizes["T"] == 1:
        arr = arr.isel(T=0, drop=True)
    else:
        arr = arr.rename({"T": "ITERATION"})
        arr = arr.assign_coords(ITERATION=np.arange(arr.sizes["ITERATION"]))

    pixel_sizes_ = img.physical_pixel_sizes
    arr = arr.rename({"C": "M_Z"})
    arr = arr.assign_coords(
        M_Z=_mz_from_channel_names(img.channel_names),
        Y=np.arange(arr.sizes["Y"]) * float(pixel_sizes_.Y or 1.0),
        X=np.arange(arr.sizes["X"]) * float(pixel_sizes_.X or 1.0),
    )

    # BioImage stashes the reader's raw metadata blobs in attrs; this
    # project keeps none of it -- everything needed is in the coords.
    arr.attrs.clear()
    arr.name = "ion_intensity"
    return arr


def _mz_from_channel_names(names: Sequence[str]) -> list[float]:
    """Float m/z per channel name: the numeric part before "±" if present
    (bioio-imzml's format), the whole name otherwise, the channel index as
    a last resort for non-numeric labels (e.g. a foreign OME-Zarr store).
    """
    values = []
    for i, name in enumerate(names):
        try:
            values.append(float(str(name).split("±")[0]))
        except ValueError:
            values.append(float(i))
    return values


def pixel_sizes(data: xr.DataArray) -> tuple[float, float]:
    """``(pixel_size_y, pixel_size_x)`` in µm, from the Y/X coord spacing --
    the only place pixel size lives in-memory (on disk it's the OME-NGFF
    axis scale, and `bioimage_to_dataarray` turns that back into coords).
    """

    def _step(dim: str) -> float:
        coords = data.coords[dim].values
        return float(coords[1] - coords[0]) if coords.size > 1 else 1.0

    return _step("Y"), _step("X")


def save_dataarray(data: xr.DataArray, path: str | Path) -> None:
    """Write an MSI DataArray to an OME-Zarr store.

    ``data`` must have ``M_Z``/``Y``/``X`` dims and may have one extra
    leading dim (e.g. ``ITERATION`` from `deconvolution.deconvolute_dataset`
    or ``PASS_NUM`` from the decay-profile notebook), which is mapped to
    the OME time axis. Only standard OME-NGFF metadata is written: m/z as
    channel labels, pixel size (from the Y/X coord spacing) as the spatial
    axis scales. The extra dim's coordinate *values* are not persisted --
    it comes back as a plain 0-based ``ITERATION`` index from
    `load_dataarray`.
    """
    missing = set(MSI_DIMS) - set(data.dims)
    if missing:
        raise ValueError(f"`data` is missing required dims: {sorted(missing)}")
    extra = [d for d in data.dims if d not in MSI_DIMS]
    if len(extra) > 1:
        raise ValueError(
            f"At most one extra dim besides {MSI_DIMS} is supported, got {extra}"
        )
    data = data.transpose(*extra, *MSI_DIMS)

    pixel_size_y, pixel_size_x = pixel_sizes(data)
    has_t = bool(extra)
    axes_names = (["t"] if has_t else []) + ["c", "y", "x"]
    axes_units = ([None] if has_t else []) + [None, "micrometer", "micrometer"]
    physical_pixel_size = ([1.0] if has_t else []) + [1.0, pixel_size_y, pixel_size_x]

    writer = OMEZarrWriter(
        store=str(path),
        level_shapes=[data.shape],
        dtype=data.dtype,
        channels=[
            Channel(label=str(float(mz)), color="FFFFFF") for mz in data.M_Z.values
        ],
        axes_names=axes_names,
        axes_units=axes_units,
        physical_pixel_size=physical_pixel_size,
    )
    writer.write_full_volume(data.data)


def load_dataarray(path: str | Path, lazy: bool = True) -> xr.DataArray:
    """Load an MSI DataArray from any BioImage-readable source -- for this
    project always an ``.ome.zarr`` store written by `save_dataarray` (see
    `bioimage_to_dataarray` for the returned convention and the `lazy`
    flag)."""
    return bioimage_to_dataarray(BioImage(path), lazy=lazy)


def parse_mz_csv(source: str | Path | bytes) -> list[float]:
    """Parse a headerless, single-column CSV of m/z values.

    ``source`` may be a path/str, or raw bytes (as returned by a marimo
    file-upload widget's ``.contents``) — anything Polars' ``read_csv``
    accepts.
    """
    df = pl.read_csv(source=source, has_header=False, new_columns=["m_z"])
    return df["m_z"].cast(pl.Float64).to_list()
