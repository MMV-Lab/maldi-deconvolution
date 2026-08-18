"""Reading imzML files, parsing m/z target lists, and persisting MSI xarray
data to disk.

``save_dataarray``/``load_dataarray`` dispatch on the path's extension:
``.zarr``/``.ome.zarr`` goes through the OME-Zarr reader/writer below
(backed by bioio-ome-zarr), anything else through xarray's built-in
netCDF/HDF5. ``M_Z`` maps to the OME channel axis and ``ITERATION`` (if
present) to the OME time axis; the OME space axes map straight through as
``Y``/``X``. All four are uppercase -- every xarray dimension/coordinate
name in this project's MSI representations is, matching bioio/BioImage's
own axis-naming convention (``T``/``C``/``Z``/``Y``/``X``).

Shared by the ``extract_data``, ``extract_decay_profile``, ``deconvolve``,
and ``visualize`` notebooks.
"""

import math
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import dask
import numpy as np
import polars as pl
import xarray as xr
import zarr
from bioio_ome_zarr.reader import Reader as OMEZarrReader
from bioio_ome_zarr.writers import (
    Channel,
    OMEZarrWriter,
    multiscale_chunk_size_from_memory_target,
)
from pyimzml.ImzMLParser import ImzMLParser
from scipy.signal import find_peaks


def read_imzml_to_xarray(
    filepath: Union[str, Path],
    m_z_values: List[float],
    tolerance: float = 0.005,
) -> xr.DataArray:
    """Read an imzML file and return an xarray DataArray of ion images.

    Parameters
    ----------
    filepath:
        Path to the .imzml file.
    m_z_values:
        List of target m/z values to extract.
    tolerance:
        Half-window (in Da) around each target m/z for intensity summation.

    Returns
    -------
    xr.DataArray
        Shape ``(n_mz, n_y, n_x)``, dims ``(M_Z, Y, X)`` (X/Y coordinates in
        μm), matching the uppercase dimension convention used throughout
        this project's xarray representations (see module docstring).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imzml = ImzMLParser(filepath)

    # --- pixel size ---
    pixel_size_x = imzml.imzmldict.get("pixel size x")
    pixel_size_y = imzml.imzmldict.get("pixel size y")
    if pixel_size_x is None or pixel_size_y is None:
        # Try to compute from max dimension and pixel count
        max_dim_x = imzml.imzmldict.get("max dimension x")
        max_dim_y = imzml.imzmldict.get("max dimension y")
        max_pix_x = imzml.imzmldict.get("max count of pixels x")
        max_pix_y = imzml.imzmldict.get("max count of pixels y")

        if not all([max_dim_x, max_dim_y, max_pix_x, max_pix_y]):
            raise ValueError(
                f"Cannot determine pixel size. Missing metadata: "
                f"max_dim_x={max_dim_x}, max_dim_y={max_dim_y}, "
                f"max_pix_x={max_pix_x}, max_pix_y={max_pix_y}"
            )
        pixel_size_x = max_dim_x / max_pix_x
        pixel_size_y = max_dim_y / max_pix_y
    pixel_size_x = float(pixel_size_x)
    pixel_size_y = float(pixel_size_y)

    # --- spatial extent (imzML coordinates are 1-based) ---
    max_x = max(c[0] for c in imzml.coordinates)
    max_y = max(c[1] for c in imzml.coordinates)

    # --- extent m_z ---

    min_mz_global = float("inf")
    max_mz_global = float("-inf")

    # Scan all spectra to determine the full ranges
    # (This iterates over all pixels, so it might take a few seconds)
    for i in range(len(imzml.coordinates)):
        _mzs, _ = imzml.getspectrum(i)
        if len(_mzs) > 0:
            # m/z arrays are typically sorted
            local_min = _mzs[0]
            local_max = _mzs[-1]

            if local_min < min_mz_global:
                min_mz_global = local_min
            if local_max > max_mz_global:
                max_mz_global = local_max

    min_mz_global = math.floor(min_mz_global)
    max_mz_global = math.ceil(max_mz_global)

    # --- validate m/z values are within range ---
    out_of_range = [
        mz for mz in m_z_values if mz < min_mz_global or mz > max_mz_global
    ]
    if out_of_range:
        raise ValueError(
            f"The following m/z values are outside the data range [{min_mz_global}, {max_mz_global}]: {out_of_range}"
        )

    # --- build ion-image stack ---
    targets = np.array(m_z_values)
    lower_bounds = targets - tolerance
    upper_bounds = targets + tolerance
    n_targets = len(targets)

    ion_image_stack = np.zeros((n_targets, max_y, max_x), dtype=np.float32)

    for idx, (x, y, _z) in enumerate(imzml.coordinates):
        x_pos, y_pos = x - 1, y - 1
        mzs, ints = imzml.getspectrum(idx)
        idx_l = np.searchsorted(mzs, lower_bounds, side="left")
        idx_r = np.searchsorted(mzs, upper_bounds, side="right")
        for t in range(n_targets):
            l, r = idx_l[t], idx_r[t]
            if r > l:
                ion_image_stack[t, y_pos, x_pos] = np.sum(ints[l:r])

    # --- wrap in xarray ---
    x_coords_um = (np.arange(max_x) + 1) * pixel_size_x
    y_coords_um = (np.arange(max_y) + 1) * pixel_size_y

    da = xr.DataArray(
        ion_image_stack,
        coords={"M_Z": m_z_values, "Y": y_coords_um, "X": x_coords_um},
        dims=("M_Z", "Y", "X"),
        attrs={
            "tolerance": tolerance,
            "pixel_size_x": pixel_size_x,
            "pixel_size_y": pixel_size_y,
            "original_file_name": Path(filepath).name,
        },
        name="ion_intensity",
    )
    da.X.attrs["units"] = "μm"
    da.Y.attrs["units"] = "μm"
    da.M_Z.attrs["long_name"] = "m/z"
    da.attrs["long_name"] = "Ion Intensity"
    da.attrs["units"] = "a.u."
    return da


def find_peaks_from_imzml(
    filepath: Union[str, Path],
    min_mz: float = 650.0,
    max_mz: float = 850.0,
    bin_width: float = 0.05,
    top_n_peaks: Optional[int] = None,
    prominence_factor: float = 5.0,
    min_peak_separation: float = 0.5,
) -> Tuple[List[float], pl.DataFrame]:
    """Build a mean spectrum histogram from an imzML file and return the top peaks.

    Parameters
    ----------
    filepath:
        Path to the .imzml file.
    min_mz, max_mz:
        m/z range to consider.
    bin_width:
        Histogram bin width in Da. Controls both the spectrum resolution
        and therefore the precision of detected peak centers.
    top_n_peaks:
        Number of highest-intensity peaks to return. Leave as ``None`` (the
        default) to return *all* detected peaks, sorted by descending
        intensity.
    prominence_factor:
        Peak prominence threshold as a multiple of the median noise floor.
    min_peak_separation:
        Minimum distance between two accepted peaks in Da.
        Converted to bins as ``max(1, int(min_peak_separation / bin_width))``.

    Returns
    -------
    target_mzs : list[float]
        Detected peak centers sorted by descending intensity, up to
        ``top_n_peaks`` (all of them if ``top_n_peaks`` is ``None``).
    spectrum : pl.DataFrame
        Mean spectrum with ``"m_z"`` (bin centers) and ``"intensity"``
        (mean intensity per bin across all pixels) columns.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser = ImzMLParser(filepath)

    all_coords = parser.coordinates
    mz_bins = np.arange(min_mz, max_mz + bin_width, bin_width)
    mz_axis = (mz_bins[:-1] + mz_bins[1:]) / 2
    mean_spectrum = np.zeros_like(mz_axis, dtype=np.float64)

    for spec_idx, _ in enumerate(all_coords):
        mzs, ints = parser.getspectrum(spec_idx)
        mask = (mzs >= min_mz) & (mzs <= max_mz)
        hist, _ = np.histogram(mzs[mask], bins=mz_bins, weights=ints[mask])
        mean_spectrum += hist

    mean_spectrum /= len(all_coords)

    noise_floor = np.median(mean_spectrum)
    distance_bins = max(1, int(min_peak_separation / bin_width))
    peak_indices, _ = find_peaks(
        mean_spectrum,
        prominence=noise_floor * prominence_factor,
        distance=distance_bins,
    )

    peak_mzs = mz_axis[peak_indices]
    peak_ints = mean_spectrum[peak_indices]
    sort_idx = np.argsort(peak_ints)[::-1]
    sorted_peak_mzs = peak_mzs[sort_idx]
    target_mzs = (
        sorted_peak_mzs if top_n_peaks is None else sorted_peak_mzs[:top_n_peaks]
    ).tolist()

    spectrum = pl.DataFrame({"m_z": mz_axis, "intensity": mean_spectrum})

    return target_mzs, spectrum


def parse_mz_csv(source: Union[str, Path, bytes]) -> List[float]:
    """Parse a headerless, single-column CSV of m/z values.

    ``source`` may be a path/str, or raw bytes (as returned by a marimo
    file-upload widget's ``.contents``) — anything Polars' ``read_csv``
    accepts.
    """
    df = pl.read_csv(source=source, has_header=False, new_columns=["m_z"])
    return df["m_z"].cast(pl.Float64).to_list()


# --- On-disk persistence -----------------------------------------------
#
# Every notebook that saves/loads a DataArray or Dataset should go through
# these four functions rather than calling `.to_netcdf`/`xr.open_dataarray`
# etc. directly. `save_dataarray`/`load_dataarray` pick netCDF/HDF5 or
# OME-Zarr based on `path`'s extension, so call sites don't need to care
# which backend actually wrote a given file.

# The two concrete extensions save_dataarray/load_dataarray accept, for
# notebooks to pass straight to `mo.ui.file_browser(filetypes=...)` instead
# of each hardcoding this list separately.
DATAARRAY_EXTENSIONS = [".h5", ".ome.zarr"]


def _is_zarr_path(path: Union[str, Path]) -> bool:
    suffixes = Path(path).suffixes
    return suffixes[-1:] == [".zarr"]


def _uppercase_dims(obj):
    """Rename any dim (DataArray) or dim/data-var (Dataset) that isn't
    already uppercase to its uppercase form -- bridges on-disk data written
    before this project's all-uppercase dim/coord convention (``M_Z``,
    ``ITERATION``, ``Y``, ``X``, and per-module dims like ``PASS_NUM``) was
    adopted, so callers never need to care which convention a given file
    predates.
    """
    rename = {d: d.upper() for d in obj.dims if d != d.upper()}
    return obj.rename(rename) if rename else obj


def _require_msi_dims(data: xr.DataArray) -> None:
    missing = {"M_Z", "Y", "X"} - set(data.dims)
    if missing:
        raise ValueError(f"`data` is missing required dims: {sorted(missing)}")


def save_dataarray(data: xr.DataArray, path: Union[str, Path]) -> None:
    """Persist a DataArray to disk: OME-Zarr (via `write_ome_zarr`) if
    `path` ends in `.zarr`/`.ome.zarr`, else netCDF/HDF5 (`.h5` filenames
    throughout this project). Requires ``M_Z``/``Y``/``X`` dims on both
    backends (see `write_ome_zarr`)."""
    _require_msi_dims(data)
    if _is_zarr_path(path):
        write_ome_zarr(data, path)
    else:
        data.to_netcdf(path)


def load_dataarray(path: Union[str, Path], lazy: bool = True) -> xr.DataArray:
    """Load a single-variable DataArray previously written by
    `save_dataarray`.

    Always returns the project's all-uppercase dim/coord convention,
    matching `read_ome_zarr`: any dim written before that convention was
    adopted is uppercased on the way in (see `_uppercase_dims`), so callers
    never need to care which convention a given `.h5` file predates.

    lazy:
        True (default) -- keeps the array disk/dask-backed until computed
            (`xr.open_dataarray` for netCDF, `read_ome_zarr(..., lazy=True)`
            for Zarr). Required before `.chunk()`-ing it for dask-parallel
            processing.
        False -- reads eagerly into memory (`xr.load_dataarray` /
            `read_ome_zarr(..., lazy=False)`). Preferable when the array
            will be sliced/indexed repeatedly afterward.
    """
    if _is_zarr_path(path):
        return read_ome_zarr(path, lazy=lazy)
    arr = xr.open_dataarray(path) if lazy else xr.load_dataarray(path)
    return _uppercase_dims(arr)


def save_dataset(data: xr.Dataset, path: Union[str, Path]) -> None:
    """Persist a Dataset to disk (always netCDF/HDF5).

    OME-Zarr support is scoped to single MSI DataArrays (see
    `write_ome_zarr`'s M_Z -> channel / ITERATION -> time mapping); this
    project's Datasets (e.g. the PSF simulation grids from
    `MaldiPSFGenerator.to_xarray`) don't have that shape, so `.zarr`/
    `.ome.zarr` paths raise rather than silently writing netCDF under a
    misleading extension.
    """
    if _is_zarr_path(path):
        raise NotImplementedError(
            "OME-Zarr is only supported for MSI DataArrays (see save_dataarray); "
            f"got a Dataset destined for {path}."
        )
    data.to_netcdf(path)


def load_dataset(path: Union[str, Path], lazy: bool = True) -> xr.Dataset:
    """Load a multi-variable Dataset previously written by `save_dataset`
    (see `load_dataarray` for the `lazy` flag and legacy-dim bridging)."""
    if _is_zarr_path(path):
        raise NotImplementedError(
            "OME-Zarr is only supported for MSI DataArrays (see load_dataarray); "
            f"got a Dataset source at {path}."
        )
    data = xr.open_dataset(path) if lazy else xr.load_dataset(path)
    return _uppercase_dims(data)


# --- OME-Zarr backing for save_dataarray/load_dataarray -----------------
#
# Maps this project's MSI DataArray convention onto OME-NGFF axes: M_Z ->
# channel axis, ITERATION (if present) -> time axis, Y/X -> space. Always
# writes Zarr v3 (NGFF 0.5) and a single resolution level (no multiscale
# pyramid) -- this is a data-interchange format for this project's own
# reader, not a file meant for external NGFF viewers, so the extra
# storage/write cost of a pyramid isn't worth it here.

# Root-group zarr attribute key holding the round-trip metadata written by
# `write_ome_zarr` (via OMEZarrWriter's `attributes=` param, merged straight
# into the root group's attrs alongside "ome"): the DataArray's name,
# original dim order, and attrs/coord_attrs. Coordinate *values*
# (M_Z/Y/X/ITERATION) aren't stored here -- they're cheaply re-derived on
# read from the store's own OME-NGFF metadata (channel labels, physical
# pixel size, array shape), same as for a foreign store.
_METADATA_KEY = "maldi_msi"

# Decimal places for the m/z channel label (e.g. "750.123456789012"). This
# is the only place m/z values are persisted on disk, so it needs enough
# precision to round-trip meaningfully -- 12 decimals is already far beyond
# any real mass-spec measurement precision.
_MZ_LABEL_DECIMALS = 12

# 16 MiB, matching bioio-ome-zarr's own chunking presets.
_DEFAULT_CHUNK_MEMORY_TARGET = 16 * 1024 * 1024

# Matplotlib "tab10" hex colors (no alpha), used as the default per-channel
# OMERO color when the caller doesn't supply `channel_colors`. Cycled if
# there are more channels than colors. The first five match the palette
# already used for decay-curve plots in `src/plotting.py`.
_DEFAULT_CHANNEL_COLORS = [
    "1F77B4",
    "FF7F0E",
    "2CA02C",
    "D62728",
    "9467BD",
    "8C564B",
    "E377C2",
    "7F7F7F",
    "BCBD22",
    "17BECF",
]


def _determine_chunk_shape(
    shape: Sequence[int],
    dtype: Union[str, np.dtype],
    axes_names: Sequence[str],
    memory_target: int,
) -> Tuple[int, ...]:
    """Pick a chunk shape favoring single-plane (one channel, one time
    point) random access.

    If a full XY plane fits under ``memory_target``, chunk by channel (and
    by time point, if present): each chunk is exactly one ``(Y, X)`` image,
    matching how the data is actually consumed downstream (picking one m/z
    at one iteration at a time in the visualize notebook). Otherwise the
    plane itself is too big to chunk that finely, so defer to
    bioio-ome-zarr's own `multiscale_chunk_size_from_memory_target`, which
    also subdivides the spatial axes.
    """
    itemsize = np.dtype(dtype).itemsize
    y_idx = axes_names.index("Y")
    x_idx = axes_names.index("X")
    plane_bytes = int(shape[y_idx]) * int(shape[x_idx]) * itemsize

    if plane_bytes <= memory_target:
        chunk = [1] * len(shape)
        chunk[y_idx] = shape[y_idx]
        chunk[x_idx] = shape[x_idx]
        return tuple(chunk)

    return tuple(
        multiscale_chunk_size_from_memory_target([tuple(shape)], dtype, memory_target)[0]
    )


def _compute_channel_windows(
    data: xr.DataArray,
    mode: Literal["minmax", "percentile"],
    percentiles: Tuple[float, float],
) -> List[Dict[str, float]]:
    """Per-channel OMERO ``window`` (contrast limits) for the channel
    metadata, one dict per M_Z value.

    "minmax" (default): min/max *and* start/end are all the channel's
    actual data extrema -- a sane default now that pixel values are
    arbitrary floats, far from the `Channel` class's own built-in default
    of a 0-255 window aimed at 8-bit images.
    "percentile": min/max stay the actual data extrema (so the full range
    is still reachable), but start/end are drawn from `percentiles` --
    robust to the hot-pixel outliers common in MS imaging, so a couple of
    saturated pixels don't wash out the default display contrast.
    """
    reduce_dims = [d for d in data.dims if d != "M_Z"]
    channel_min = data.min(dim=reduce_dims)
    channel_max = data.max(dim=reduce_dims)

    if mode == "percentile":
        lo_q, hi_q = percentiles[0] / 100.0, percentiles[1] / 100.0
        channel_quantiles = data.quantile([lo_q, hi_q], dim=reduce_dims)
        channel_min, channel_max, channel_quantiles = dask.compute(
            channel_min, channel_max, channel_quantiles
        )
        starts = channel_quantiles.isel(quantile=0).values
        ends = channel_quantiles.isel(quantile=1).values
    else:
        channel_min, channel_max = dask.compute(channel_min, channel_max)
        starts = channel_min.values
        ends = channel_max.values

    return [
        {"min": float(lo), "max": float(hi), "start": float(s), "end": float(e)}
        for lo, hi, s, e in zip(channel_min.values, channel_max.values, starts, ends)
    ]


def write_ome_zarr(
    data: xr.DataArray,
    path: Union[str, Path],
    *,
    channel_colors: Optional[Sequence[str]] = None,
    window_mode: Literal["minmax", "percentile"] = "minmax",
    window_percentiles: Tuple[float, float] = (1.0, 99.0),
    chunk_memory_target: int = _DEFAULT_CHUNK_MEMORY_TARGET,
) -> None:
    """Write an MSI DataArray to an OME-Zarr (Zarr v3 / NGFF 0.5) store.

    ``data`` must have ``M_Z``, ``Y``, ``X`` dims, and may optionally have
    an ``ITERATION`` dim (as produced by
    `deconvolution.deconvolute_dataset`). ``M_Z`` is mapped to the OME
    channel axis and ``ITERATION`` to the OME time axis.

    Only a single resolution level is written (no multiscale pyramid).

    channel_colors:
        Hex color strings (e.g. ``"FF0000"``, no alpha channel), one per
        channel, cycled if shorter than the number of m/z channels.
        Defaults to `_DEFAULT_CHANNEL_COLORS` (matplotlib's "tab10").
    window_mode, window_percentiles:
        See `_compute_channel_windows`.
    """
    if window_mode not in ("minmax", "percentile"):
        raise ValueError(
            f"window_mode must be 'minmax' or 'percentile', got {window_mode!r}"
        )
    _require_msi_dims(data)

    has_iteration = "ITERATION" in data.dims
    original_dims = list(data.dims)
    dims_order = (("ITERATION",) if has_iteration else ()) + ("M_Z", "Y", "X")
    data = data.transpose(*dims_order)

    axes_names = (["t"] if has_iteration else []) + ["c", "Y", "X"]
    axes_types = (["time"] if has_iteration else []) + ["channel", "space", "space"]
    axes_units = ([None] if has_iteration else []) + [None, "micrometer", "micrometer"]
    physical_pixel_size = ([1.0] if has_iteration else []) + [
        1.0,
        float(data.attrs.get("pixel_size_y", 1.0)),
        float(data.attrs.get("pixel_size_x", 1.0)),
    ]

    colors = list(channel_colors) if channel_colors else _DEFAULT_CHANNEL_COLORS
    windows = _compute_channel_windows(data, window_mode, window_percentiles)
    channels = [
        Channel(
            label=f"{mz:.{_MZ_LABEL_DECIMALS}f}",
            color=colors[i % len(colors)],
            window=windows[i],
        )
        for i, mz in enumerate(data.M_Z.values)
    ]

    shape = data.shape
    chunk_shape = _determine_chunk_shape(shape, data.dtype, axes_names, chunk_memory_target)
    image_name = str(data.attrs.get("original_file_name") or data.name or "Image")

    coord_dims = ("M_Z", "Y", "X") + (("ITERATION",) if has_iteration else ())
    attributes = {
        _METADATA_KEY: {
            "name": data.name,
            "dims": original_dims,
            "coord_attrs": {dim: dict(data.coords[dim].attrs) for dim in coord_dims},
            "attrs": dict(data.attrs),
        }
    }

    writer = OMEZarrWriter(
        store=str(path),
        level_shapes=[shape],
        dtype=data.dtype,
        chunk_shape=chunk_shape,
        zarr_format=3,
        image_name=image_name,
        channels=channels,
        attributes=attributes,
        axes_names=axes_names,
        axes_types=axes_types,
        axes_units=axes_units,
        physical_pixel_size=physical_pixel_size,
    )
    writer.write_full_volume(data.data)


def _parse_float_or(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def read_ome_zarr(path: Union[str, Path], lazy: bool = True) -> xr.DataArray:
    """Read an MSI DataArray from an OME-Zarr store.

    Coordinates are always re-derived from the store's own OME-NGFF
    metadata, whether or not it was written by `write_ome_zarr`: M_Z from
    the channel labels (parsed as float; falls back to a 0-based index for
    a non-numeric label), Y/X from the physical pixel size and array shape
    (matching `read_imzml_to_xarray`'s ``(arange(n) + 1) * pixel_size``
    convention), and ITERATION (if a time axis is present) as a plain
    0..T-1 index. None of that needs to be duplicated on disk.

    Every dim/coord name comes back uppercase (``M_Z``, ``ITERATION``,
    ``Y``, ``X``), matching this project's convention throughout (see
    module docstring) as well as the Reader's native (always uppercased)
    axis names for Y/X.

    What *can't* be re-derived -- the DataArray's name, original dim order,
    and attrs/coord_attrs (e.g. ``tolerance``, ``original_file_name``) --
    is read back from the round-trip metadata `write_ome_zarr` stashes via
    `OMEZarrWriter`'s ``attributes=`` param (merged into the root group's
    attrs). Falls back to generic defaults for Zarr stores written by other
    tools, which won't have it.

    lazy:
        True (default) -- dask-backed, matching `load_dataarray`.
        False -- read fully into memory.
    """
    root = zarr.open_group(str(path), mode="r")
    meta = root.attrs.get(_METADATA_KEY)

    reader = OMEZarrReader(str(path))
    arr = reader.xarray_dask_data if lazy else reader.xarray_data

    rename = {}
    if "C" in arr.dims:
        rename["C"] = "M_Z"
    if "T" in arr.dims:
        rename["T"] = "ITERATION"
    arr = arr.rename(rename) if rename else arr

    pixel_sizes = reader.physical_pixel_sizes
    pixel_size_x = pixel_sizes.X if pixel_sizes.X is not None else 1.0
    pixel_size_y = pixel_sizes.Y if pixel_sizes.Y is not None else 1.0

    coords = {
        "M_Z": [
            _parse_float_or(label, default=i)
            for i, label in enumerate(reader.channel_names)
        ],
        "Y": (np.arange(arr.sizes["Y"]) + 1) * pixel_size_y,
        "X": (np.arange(arr.sizes["X"]) + 1) * pixel_size_x,
    }
    if "ITERATION" in arr.dims:
        coords["ITERATION"] = np.arange(arr.sizes["ITERATION"])
    arr = arr.assign_coords(coords)

    if meta is not None:
        # `meta["dims"]`/`meta["coord_attrs"]` keys are whatever
        # `write_ome_zarr` stashed verbatim -- uppercase them too, so a
        # store written before this project's all-uppercase convention was
        # adopted (e.g. dims stashed as ["m_z", "y", "x"]) still round-trips
        # against `arr`'s now-always-uppercase dims.
        arr = arr.transpose(*(d.upper() for d in meta["dims"]))
        arr.name = meta["name"]
        arr.attrs = dict(meta["attrs"])
        for dim, coord_attrs in meta["coord_attrs"].items():
            arr.coords[dim.upper()].attrs = dict(coord_attrs)
        return arr

    # No `write_ome_zarr` metadata (a Zarr store written by another tool) --
    # generic defaults matching this project's usual attrs.
    arr.attrs = {
        "pixel_size_x": float(pixel_size_x),
        "pixel_size_y": float(pixel_size_y),
        "original_file_name": Path(path).name,
    }
    arr.X.attrs["units"] = "μm"
    arr.Y.attrs["units"] = "μm"
    arr.M_Z.attrs["long_name"] = "m/z"
    arr.name = arr.name or "ion_intensity"

    return arr
