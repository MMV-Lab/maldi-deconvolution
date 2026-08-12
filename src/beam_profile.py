"""Loading `.cma` beam-profile scans and fitting 2D Gaussians to them,
shared by the ``beamprofile`` notebook.
"""

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import polars as pl
import xarray as xr
from astropy.modeling import fitting, models
from astropy.modeling.functional_models import Gaussian2D


def load_cma_images(
    path_list: Sequence[Union[str, Path]], x_pixel_size: float, y_pixel_size: float
) -> xr.DataArray:
    """Read a set of `.cma` CSV scans into a ``(SAMPLE, Y, X)`` intensity
    stack with physical (µm) coordinates.

    Each CSV has a trailing empty column, which is dropped.
    """
    images = [
        pl.read_csv(p, has_header=False).to_numpy()[:, :-1].astype(float)
        for p in path_list
    ]
    stacked_data = np.stack(images)

    return xr.DataArray(
        stacked_data,
        dims=["SAMPLE", "Y", "X"],
        coords={
            "SAMPLE": np.arange(stacked_data.shape[0]),
            "Y": np.arange(stacked_data.shape[1]) * y_pixel_size,
            "X": np.arange(stacked_data.shape[2]) * x_pixel_size,
        },
        name="intensity",
        attrs={"units": "um"},
    )


def crop_around_peak(xy_profiles: xr.DataArray, window: int = 20) -> xr.DataArray:
    """Crop a ``(SAMPLE, Y, X)`` stack to a ``window`` x ``window`` box
    centered on the sample-averaged peak position."""
    max_pos = xy_profiles.mean(dim="SAMPLE").argmax(dim=["Y", "X"])
    half = window // 2
    y_mid = int(max_pos["Y"])
    x_mid = int(max_pos["X"])

    return xy_profiles.isel(
        Y=slice(y_mid - half, y_mid + half),
        X=slice(x_mid - half, x_mid + half),
    )


def fit_gaussian2d(
    x_coords: np.ndarray, y_coords: np.ndarray, z_data: np.ndarray
) -> Gaussian2D:
    """Fit a 2D Gaussian (astropy.modeling) to ``z_data``, shape
    ``(len(y_coords), len(x_coords))``, initialized at its peak pixel.
    """
    X, Y = np.meshgrid(x_coords, y_coords)

    max_idx = np.unravel_index(np.argmax(z_data), z_data.shape)
    y_peak_guess = y_coords[max_idx[0]]
    x_peak_guess = x_coords[max_idx[1]]

    model_init = models.Gaussian2D(
        amplitude=z_data.max(),
        x_mean=x_peak_guess,
        y_mean=y_peak_guess,
        x_stddev=1.0,
        y_stddev=1.0,
    )

    fitter = fitting.LevMarLSQFitter()
    return fitter(model_init, X, Y, z_data)


def beam_diameter_stats(
    fit_results: List[dict],
) -> Tuple[pl.DataFrame, Dict[str, float]]:
    """Combine per-sample Gaussian fits into a table with derived 1/e^2
    beam diameters, plus mean/std/sem summary statistics.

    Returns
    -------
    df : pl.DataFrame
        ``fit_results`` plus ``sigma_eff`` and ``beam_diameter_1e2`` columns.
    summary : dict
        ``{"mean": ..., "std": ..., "sem": ...}`` for ``beam_diameter_1e2``.
    """
    df = (
        pl.DataFrame(fit_results)
        .with_columns(sigma_eff=(pl.col("x_stddev") * pl.col("y_stddev")).sqrt())
        .with_columns(beam_diameter_1e2=pl.col("sigma_eff") * 4)
    )

    mean = df["beam_diameter_1e2"].mean()
    std = df["beam_diameter_1e2"].std()
    sem = std / math.sqrt(df["beam_diameter_1e2"].count())

    return df, {"mean": mean, "std": std, "sem": sem}
