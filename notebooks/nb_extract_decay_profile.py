import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import numpy as np
    from pyimzml.ImzMLParser import ImzMLParser
    import warnings
    import re

    import polars as pl
    import xarray as xr

    import holoviews as hv
    import hvplot
    import hvplot.xarray
    import hvplot.polars

    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    from skimage.filters import threshold_li, threshold_triangle
    import scipy.stats as stats

    from extract_data import read_imzml_to_xarray

    hvplot.extension("matplotlib", "bokeh")
    hvplot.output(fig="svg")

    hv.extension('matplotlib')
    hv.output(fig='svg')


@app.cell
def _():
    hvplot.output.info()
    return


@app.cell
def _():
    mo.md(r"""
    # Welcome!

    ## This notebook serves the purpose of extracting the decay profile from multiple `imzML` files (over repeated scans)
    """)
    return


@app.cell
def _():
    widget_selected_file = mo.ui.file_browser(
        filetypes=[".imzml"], multiple=True, initial_path="data"
    )
    mo.vstack([mo.md("# Select `imzml` file"), widget_selected_file])
    return (widget_selected_file,)


@app.cell
def _(widget_selected_file):
    path_list = [i.path for i in widget_selected_file.value]
    path_list = sorted(
        path_list, key=lambda x: int(re.search(r"(\d+)p(?:ass)?", x.name).group(1))
    )

    [i.name for i in path_list]
    return (path_list,)


@app.cell
def _(path_list):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imzml_list = [ImzMLParser(i) for i in path_list]
    return (imzml_list,)


@app.function
def find_peaks_from_imzml(
    filepath,
    min_mz: float = 650.0,
    max_mz: float = 850.0,
    bin_width: float = 0.05,
    top_n_peaks: int = 10,
    prominence_factor: float = 5.0,
    min_peak_separation: float = 0.5,
):
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
        Number of highest-intensity peaks to return.
    prominence_factor:
        Peak prominence threshold as a multiple of the median noise floor.
    min_peak_separation:
        Minimum distance between two accepted peaks in Da.
        Converted to bins as ``max(1, int(min_peak_separation / bin_width))``.

    Returns
    -------
    target_mzs : list[float]
        Detected peak centers sorted by descending intensity, up to ``top_n_peaks``.
    mz_axis : np.ndarray
        Bin centers of the mean spectrum.
    mean_spectrum : np.ndarray
        Mean intensity per bin across all pixels.
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
    target_mzs = peak_mzs[sort_idx][:top_n_peaks].tolist()

    return target_mzs, mz_axis, mean_spectrum


@app.cell
def _():
    return


@app.cell
def _(imzml_list, path_list):
    # ==========================================
    # CONFIGURATION
    # ==========================================
    min_mz, max_mz = 650.0, 850.0
    mz_tol = 0.5  # Da window around each peak for extraction
    top_n_peaks = 10  # How many automatic peaks to track
    bin_width = 0.05

    n_passes = len(imzml_list)
    parser_pass1 = imzml_list[0]
    all_coords = parser_pass1.coordinates
    max_x = max(c[0] for c in all_coords)
    max_y = max(c[1] for c in all_coords)

    # ==========================================
    # AUTOMATIC PEAK FINDING (on Pass 1)
    # ==========================================

    with mo.status.progress_bar(
        total=len(all_coords),
        title="Step 1: Automatic Peak Finding",
        subtitle="Building Mean Spectrum of Pass 1...",
    ) as bar:
        target_mzs, mz_axis, mean_spectrum = find_peaks_from_imzml(
            path_list[0],
            min_mz=min_mz,
            max_mz=max_mz,
            bin_width=bin_width,
            top_n_peaks=top_n_peaks,
            min_peak_separation=1.5 * mz_tol,
        )
        bar.update(len(all_coords))

    mo.output.append(
        mo.md(
            f"**🎯 Automatically Detected Top {len(target_mzs)} Peaks:** `{np.round(target_mzs, 3).tolist()}`"
        )
    )

    # target_mzs = [760.5749999998994, 798.5249999998649, 734.5749999999231, 772.5249999998886, 788.6249999998739, 761.5749999998985, 808.5749999998558, 799.524999999864, 826.5749999998394, 735.5749999999222]


    # ==========================================
    # BUILD 4D DATA CUBE (pass, mz, y, x)
    # ==========================================
    with mo.status.progress_bar(
        total=n_passes,
        title="Step 2: Building 4D Xarray Data Cube",
        subtitle="Extracting peaks individually...",
    ) as bar:
        _pass_arrays = []
        for filepath_p in path_list:
            _da = read_imzml_to_xarray(filepath_p, target_mzs, mz_tol)
            _pass_arrays.append(_da)
            bar.update()

    passes_xr = xr.concat(
        _pass_arrays,
        dim=xr.DataArray(
            np.arange(1, n_passes + 1), dims="pass_num", name="pass_num"
        ),
    )
    passes_xr.name = "individual_peak_intensities"

    # ==========================================
    # MASK GENERATION
    # ==========================================

    pass1 = passes_xr.sel(pass_num=1)
    pass1_clipped = pass1.clip(
        max=pass1.quantile(0.9, dim=["y", "x"], skipna=True)
    ).drop_vars("quantile")

    thresholds = xr.apply_ufunc(
        # threshold_triangle,
        threshold_li,
        # pass1,
        pass1_clipped,
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
    return (
        masked_passes,
        mean_spectrum,
        mz_axis,
        n_passes,
        passes_xr,
        stats_ds,
        target_mzs,
    )


@app.cell
def _(passes_xr):
    passes_xr
    return


@app.cell
def _():
    # tmp = xr.open_dataarray("data/tmp.h5")
    # tmp
    return


@app.cell
def _(mean_spectrum, mz_axis, target_mzs):
    _spectrum_df = pl.DataFrame({"m_z": mz_axis, "intensity": mean_spectrum})
    _peaks_df = pl.DataFrame(
        {
            "m_z": target_mzs,
            "intensity": [
                mean_spectrum[
                    int(round((mz - mz_axis[0]) / (mz_axis[1] - mz_axis[0])))
                ]
                for mz in target_mzs
            ],
        }
    )

    _spectrum_plot = _spectrum_df.hvplot.line(
        x="m_z",
        y="intensity",
        label="Mean spectrum",
        color="steelblue",
        alpha=0.8,
        xlabel="m/z (Da)",
        ylabel="Mean intensity (a.u.)",
        title="Mean spectrum — Pass 1 (detected peaks marked)",
    )
    _peaks_plot = _peaks_df.hvplot.scatter(
        x="m_z",
        y="intensity",
        label="Detected peaks",
        color="red",
        size=120,
    )
    _spectrum_plot * _peaks_plot
    return


@app.cell
def _():
    _spectrum_plot * _peaks_plot
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

    show_export_button = True

    decay_df = pl.from_pandas(norm_ds.to_dataframe().reset_index())
    del norm_ds
    decay_df
    return decay_df, show_export_button


@app.cell
def _(decay_df):
    (
        decay_df.hvplot.errorbars(
            x="pass_num", y="value", yerr1="error", width=400
        )
        * decay_df.hvplot.scatter(x="pass_num", y="value", s=800)
    )
    return


@app.cell
def _(show_export_button):
    export_button = mo.ui.run_button(label="Export")

    mo.stop(not show_export_button)
    mo.vstack(
        [
            mo.md("### Press the following button to save the decay profile:"),
            export_button,
        ]
    )
    return (export_button,)


@app.cell
def _(decay_df, export_button, passes_xr, path_list):
    mo.stop(not export_button.value)

    _out_path = path_list[0].parent / "Decay_Profile.csv"

    decay_df.write_csv(_out_path, float_scientific=False, float_precision=16)

    passes_xr.to_netcdf(
        path_list[0].parent / f"multiple_passes_filtered.h5",
    )

    mo.md(f"""Saved decay profile as `{_out_path.name}` in:

    `{_out_path.parent}`

    """).callout(kind="success")
    return


@app.cell
def _(masked_passes, passes_xr):
    (
        passes_xr.hvplot.image(x="x", y="y", aspect="square", cmap="magma")
        + masked_passes.hvplot.image(x="x", y="y", aspect="square", cmap="magma")
    )
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
    return fit_ds, results_dict


@app.cell
def _():
    hvplot.output.info()
    return


@app.cell
def _(results_dict):
    results_dict
    return


@app.cell
def _(decay_df, fit_ds):
    error_bars = decay_df.hvplot.errorbars(
        x="pass_num",
        y="value",
        yerr1="error",
        color="black",
        alpha=0.4,
        width=400,
    )

    scatter_plot = decay_df.hvplot.scatter(
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
