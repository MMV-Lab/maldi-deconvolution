"""Tissue masking, per-pass statistics, and decay-curve model fitting shared
by the ``extract_decay_profile`` notebook.
"""

from typing import Sequence, Union

import numpy as np
import scipy.stats as stats
import xarray as xr
from scipy.optimize import curve_fit
from skimage.filters import threshold_li

FloatOrArray = Union[float, np.ndarray]


def compute_tissue_mask(
    pass1: xr.DataArray, clip_quantile: float = 0.9
) -> xr.DataArray:
    """Threshold the first pass of a multi-pass m/z stack into a tissue mask.

    The image is clipped at ``clip_quantile`` before Li thresholding so a
    few very bright outlier pixels don't skew the threshold.
    """
    pass1_clipped = pass1.clip(
        max=pass1.quantile(clip_quantile, dim=["y", "x"], skipna=True)
    ).drop_vars("quantile")

    thresholds = xr.apply_ufunc(
        threshold_li,
        pass1_clipped,
        input_core_dims=[["y", "x"]],
        vectorize=True,
    )

    mz_masks = pass1 > thresholds
    mz_masks.name = "tissue_mask"
    return mz_masks


def compute_pass_statistics(
    masked_passes: xr.DataArray, spatial_dims: Sequence[str] = ("y", "x")
) -> xr.Dataset:
    """Per-pass summary statistics (mean, median, spread, ...) over the
    tissue-masked region, one value per ``pass_num`` (and any other
    remaining dims, e.g. ``m_z``)."""
    spatial_dims = list(spatial_dims)

    stats_ds = xr.Dataset(
        {
            "count": masked_passes.count(dim=spatial_dims),
            "mean": masked_passes.mean(dim=spatial_dims, skipna=True),
            "median": masked_passes.median(dim=spatial_dims, skipna=True),
            "std": masked_passes.std(dim=spatial_dims, skipna=True, ddof=1),
            "var": masked_passes.var(dim=spatial_dims, skipna=True),
            "min": masked_passes.min(dim=spatial_dims, skipna=True),
            "max": masked_passes.max(dim=spatial_dims, skipna=True),
            "sum": masked_passes.sum(dim=spatial_dims, skipna=True),
            "q25": masked_passes.quantile(
                0.25, dim=spatial_dims, skipna=True
            ).drop_vars("quantile"),
            "q75": masked_passes.quantile(
                0.75, dim=spatial_dims, skipna=True
            ).drop_vars("quantile"),
        }
    )

    stats_ds["mad"] = xr.apply_ufunc(
        lambda x: stats.median_abs_deviation(
            x, axis=(-2, -1), nan_policy="omit", scale="normal"
        ),
        masked_passes,
        input_core_dims=[spatial_dims],
        vectorize=True,
        dask="parallelized",
    )

    stats_ds["iqr"] = stats_ds["q75"] - stats_ds["q25"]
    stats_ds["sem"] = stats_ds["std"] / np.sqrt(stats_ds["count"])
    stats_ds["se_median"] = stats_ds["sem"] * 1.2533

    return stats_ds


def model_exp(x: FloatOrArray, a: float, b: float, c: float) -> FloatOrArray:
    return a * np.exp(-b * x) + c


def model_biexp(
    x: FloatOrArray, a: float, b: float, c: float, d: float, e: float
) -> FloatOrArray:
    return a * np.exp(-b * x) + c * np.exp(-d * x) + e


def model_power(x: FloatOrArray, a: float, b: float, c: float) -> FloatOrArray:
    return a * np.power(x, -b) + c


def model_sigmoid(
    x: FloatOrArray, L: float, k: float, x0: float, b: float
) -> FloatOrArray:
    return L / (1 + np.exp(k * (x - x0))) + b


def fit_decay_models(
    x_data: np.ndarray, y_data: np.ndarray, y_weights: np.ndarray, n_passes: int
) -> xr.Dataset:
    """Weighted least-squares fit of the four candidate decay models.

    Returns
    -------
    fit_ds : xr.Dataset
        Fit curves (``y_fit`` over ``x_fit``) and residual metrics (``mse``,
        ``chi_sq_red``, ``params``) per model.
    """
    model_definitions = {
        "Single Exp": (model_exp, [1, 0.5, 0]),
        "Bi-Exp": (model_biexp, [0.5, 1.0, 0.5, 0.1, 0]),
        "Power Law": (model_power, [1, 0.5, 0]),
        "Sigmoid": (model_sigmoid, [1, 1, n_passes / 2, 0]),
    }

    x_fit = np.linspace(1, n_passes, 100)

    results_dict = {}
    # Find the maximum number of parameters to allow for padding in the parameters array
    max_params = max(len(p0) for _, (_, p0) in model_definitions.items())

    for name, (func, p0) in model_definitions.items():
        try:
            popt, _ = curve_fit(
                func,
                x_data,
                y_data,
                p0=p0,
                sigma=y_weights,
                absolute_sigma=True,
                maxfev=10000,
            )

            # 1. Predictions for the fit curve
            y_fit_vals = func(x_fit, *popt)

            # 2. Metrics
            res = y_data - func(x_data, *popt)
            chi_sq = np.sum((res / y_weights) ** 2)
            red_chi_sq = chi_sq / (len(x_data) - len(p0))
            mse = np.mean(res**2)

            # 3. Store results in a nested dict to build the Dataset later
            # We pad parameters with NaN if they have different lengths
            padded_params = np.full(max_params, np.nan)
            padded_params[: len(popt)] = popt

            results_dict[name] = {
                "y_fit": y_fit_vals,
                "weighted_mse": mse,
                "reduced_chi_sq": red_chi_sq,
                "params": padded_params,
            }

        except Exception as e:
            print(f"Fit failed for {name}: {e}")

    model_names = list(results_dict.keys())

    fit_ds = xr.Dataset(
        data_vars={
            "y_fit": (
                ["model", "x_fit"],
                [results_dict[m]["y_fit"] for m in model_names],
            ),
            "mse": (
                ["model"],
                [results_dict[m]["weighted_mse"] for m in model_names],
            ),
            "chi_sq_red": (
                ["model"],
                [results_dict[m]["reduced_chi_sq"] for m in model_names],
            ),
            "params": (
                ["model", "param_idx"],
                [results_dict[m]["params"] for m in model_names],
            ),
        },
        coords={
            "model": model_names,
            "x_fit": x_fit,
            "param_idx": np.arange(max_params),
        },
        attrs={"description": "Curve fitting results for multiple models"},
    )
    return fit_ds
