import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    from pyimzml.ImzMLParser import ImzMLParser, getionimage, browse
    import hvplot
    import xarray as xr
    import hvplot.xarray
    import hvplot.polars
    import hvplot.pandas
    import warnings
    import polars as pl
    import tifffile
    import datetime
    import re

    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    from skimage.filters import threshold_li, threshold_triangle
    import scipy.stats as stats

    hvplot.extension("matplotlib", "bokeh")


@app.cell
def _():
    mo.md(r"""
    # Welcome!

    ## This notebook serves the purpose of extracting data from a `imzML` file to a `h5` file which can be processed later (deconvolution or visualization)
    """)
    return


@app.cell
def _():
    widget_selected_file = mo.ui.file_browser(filetypes=[".imzml"], multiple=True)
    mo.vstack([mo.md("# Select `imzml` file"), widget_selected_file])
    return (widget_selected_file,)


@app.cell
def _(widget_selected_file):
    path_list = [i.path for i in widget_selected_file.value]
    path_list = sorted(
        path_list, key=lambda x: int(re.search(r"(\d+)pass", x.name).group(1))
    )
    return (path_list,)


@app.cell
def _(path_list):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imzml_list = [ImzMLParser(i) for i in path_list]
    return (imzml_list,)


@app.cell
def _(imzml_list):
    # ==========================================
    # CONFIGURATION
    # ==========================================
    min_mz, max_mz = 650.0, 850.0
    mz_tol = 0.5  # Da window around each peak for extraction
    top_n_peaks = 10  # How many automatic peaks to track
    saturation_limit = np.inf  # Set to detector limit if known (e.g., 65535)
    bin_width = 0.05

    n_passes = len(imzml_list)
    parser_pass1 = imzml_list[0]
    all_coords = parser_pass1.coordinates
    max_x = max(c[0] for c in all_coords)
    max_y = max(c[1] for c in all_coords)

    # ==========================================
    # AUTOMATIC PEAK FINDING (on Pass 1)
    # ==========================================

    mz_bins = np.arange(min_mz, max_mz + bin_width, bin_width)
    mz_axis = (mz_bins[:-1] + mz_bins[1:]) / 2
    mean_spectrum = np.zeros_like(mz_axis, dtype=np.float64)

    with mo.status.progress_bar(
        total=len(all_coords),
        title="Step 1: Automatic Peak Finding",
        subtitle="Building Mean Spectrum of Pass 1...",
    ) as bar:
        for spec_idx, _ in enumerate(all_coords):
            mzs, ints = parser_pass1.getspectrum(spec_idx)
            mask = (mzs >= min_mz) & (mzs <= max_mz)

            # Calculate histogram efficiently
            hist, _ = np.histogram(mzs[mask], bins=mz_bins, weights=ints[mask])
            mean_spectrum += hist
            bar.update()

    mean_spectrum /= len(all_coords)

    # Find peaks dynamically
    noise_floor = np.median(mean_spectrum)
    peak_indices, _ = find_peaks(
        mean_spectrum, prominence=noise_floor * 5, distance=int(0.5 / bin_width)
    )

    peak_mzs = mz_axis[peak_indices]
    peak_ints = mean_spectrum[peak_indices]

    sort_idx = np.argsort(peak_ints)[::-1]
    target_mzs = peak_mzs[sort_idx][:top_n_peaks].tolist()

    mo.output.append(
        mo.md(
            f"**🎯 Automatically Detected Top {len(target_mzs)} Peaks:** `{np.round(target_mzs, 3).tolist()}`"
        )
    )

    # ==========================================
    # BUILD 4D DATA CUBE (pass, mz, y, x)
    # ==========================================
    data_cube = np.zeros(
        (n_passes, len(target_mzs), max_y, max_x), dtype=np.float32
    )

    target_bounds = [(mz - mz_tol, mz + mz_tol) for mz in target_mzs]

    with mo.status.progress_bar(
        total=n_passes,
        title="Step 2: Building 4D Xarray Data Cube",
        subtitle="Extracting peaks individually...",
    ) as bar:
        for p_idx, parser in enumerate(imzml_list):
            for spec_idx, (x, y, z) in enumerate(parser.coordinates):
                mzs, ints = parser.getspectrum(spec_idx)

                for mz_idx, (low, high) in enumerate(target_bounds):
                    idx_l = np.searchsorted(mzs, low, side="left")
                    idx_r = np.searchsorted(mzs, high, side="right")

                    if idx_r > idx_l:
                        data_cube[p_idx, mz_idx, y - 1, x - 1] = np.sum(
                            ints[idx_l:idx_r]
                        )

            bar.update()

    passes_xr = xr.DataArray(
        data_cube,
        coords={
            "pass_num": np.arange(1, n_passes + 1),
            "m_z": target_mzs,
            "y": np.arange(1, max_y + 1),
            "x": np.arange(1, max_x + 1),
        },
        dims=["pass_num", "m_z", "y", "x"],
        name="individual_peak_intensities",
    )


    # ==========================================
    # MASK GENERATION
    # ==========================================

    pass1 = passes_xr.sel(pass_num=1)
    pass1_clipped = pass1.clip(
        max=pass1.quantile(0.9, dim=["y", "x"], skipna=True)
    ).drop_vars("quantile")

    thresholds = xr.apply_ufunc(
        threshold_triangle,
        # threshold_li,
        pass1,
        # pass1_clipped,
        input_core_dims=[["y", "x"]],  # Core dimensions the function needs
        vectorize=True,
    )

    mz_masks = pass1 > thresholds
    mz_masks.name = "tissue_mask"


    # ==========================================================================

    masked_passes = passes_xr.where(mz_masks)

    spatial_dims = ["y", "x"]

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
    return masked_passes, n_passes, stats_ds


@app.cell
def _(stats_ds):
    stats_ds.to_dataframe()
    return


@app.cell
def _(stats_ds):
    value = "median"
    error = "std"

    norm_ds = xr.Dataset(
        {
            "value": stats_ds[value] / stats_ds[value].sel(pass_num=1),
            "error": stats_ds[error] / stats_ds[value].sel(pass_num=1),
        }
    )

    norm_ds = norm_ds.mean(dim="m_z")
    return (norm_ds,)


@app.cell
def _(norm_ds):
    (
        norm_ds.hvplot.errorbars(x="pass_num", y="value", yerr1="error", width=400)
        * norm_ds.hvplot.scatter(x="pass_num", y="value", s=800)
    )
    return


@app.cell
def _(norm_ds):
    decay_df = pl.from_pandas(norm_ds.to_dataframe().reset_index())
    decay_df
    return (decay_df,)


@app.cell
def _(masked_passes):
    masked_passes.hvplot.image(aspect="square", cmap="magma")
    return


@app.cell
def _(decay_df, n_passes):
    # ==========================================
    # DEFINE MODELS
    # ==========================================
    def model_exp(x, a, b, c):
        return a * np.exp(-b * x) + c


    def model_biexp(x, a, b, c, d, e):
        return a * np.exp(-b * x) + c * np.exp(-d * x) + e


    def model_power(x, a, b, c):
        return a * np.power(x, -b) + c


    def model_sigmoid(x, L, k, x0, b):
        return L / (1 + np.exp(k * (x - x0))) + b


    model_definitions = {
        "Single Exp": (model_exp, [1, 0.5, 0]),
        "Bi-Exp": (model_biexp, [0.5, 1.0, 0.5, 0.1, 0]),
        "Power Law": (model_power, [1, 0.5, 0]),
        "Sigmoid": (model_sigmoid, [1, 1, n_passes / 2, 0]),
    }

    # ==========================================
    # PREPARE DATA
    # ==========================================
    x_data = decay_df["pass_num"].to_numpy()
    y_data = decay_df["value"].to_numpy()
    y_weights = decay_df["error"].to_numpy()

    x_fit = np.linspace(1, n_passes, 100)

    # ==========================================
    # WEIGHTED FITTING
    # ==========================================

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
    return (fit_ds,)


@app.cell
def _(decay_df, fit_ds):
    df_pd = decay_df.to_pandas()


    error_bars = df_pd.hvplot.errorbars(
        x="pass_num",
        y="value",
        yerr1="error",
        color="black",
        alpha=0.4,
        width=400,
    )

    scatter_plot = df_pd.hvplot.scatter(
        x="pass_num",
        y="value",
        label="Observed (w/ Std Dev)",
        c="black",
        size=500,
        alpha=0.6,
        title="Weighted Decay Model Fitting",
    )


    fit_plots = fit_ds.drop_dims("param_idx").hvplot.line(
        x="x_fit", y="y_fit", by="model", alpha=0.4
    )

    # 4. Final Overlay (Order: Scatter * ErrorBars * Fits)
    final_plot = (error_bars * fit_plots * scatter_plot).opts(
        xlabel="Pass Number",
        ylabel="Normalized Intensity (Ratio)",
        legend_position="best",
        show_grid=True,
    )

    mo.output.append(final_plot)
    mo.output.append(mo.md("### Weighted Model Accuracy (Reduced Chi-Square)"))
    mo.output.append(fit_ds[["mse", "chi_sq_red"]].to_dataframe())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
