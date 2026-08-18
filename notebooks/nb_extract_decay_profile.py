import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import numpy as np
    import re

    import polars as pl
    import xarray as xr

    import holoviews as hv
    import hvplot
    import hvplot.xarray
    import hvplot.polars

    import sys
    from pathlib import Path

    _root_dir = Path(__file__).resolve().parent.parent
    if str(_root_dir) not in sys.path:
        sys.path.append(str(_root_dir))

    from bioio_imzml import auto_pick_peaks

    from src.imzml_io import read_imzml_to_xarray, save_dataarray
    from src.decay_profile import (
        compute_tissue_mask,
        compute_pass_statistics,
        fit_decay_models,
    )
    from src.plotting import register_hv_backends, set_hv_backend

    register_hv_backends("matplotlib", "bokeh")
    set_hv_backend("matplotlib")


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
def _():
    return


@app.cell
def _(path_list):
    # ==========================================
    # CONFIGURATION
    # ==========================================
    min_mz, max_mz = 650.0, 850.0
    mz_tol = 0.5  # Da window around each peak for extraction
    top_n_peaks = 10  # How many automatic peaks to track
    bin_width = 0.05

    n_passes = len(path_list)

    # ==========================================
    # AUTOMATIC PEAK FINDING (on Pass 1)
    # ==========================================

    with mo.status.spinner(
        title="Step 1: Automatic Peak Finding",
        subtitle="Building Mean Spectrum of Pass 1...",
    ):
        _peaks = auto_pick_peaks(
            path_list[0],
            min_mz=min_mz,
            max_mz=max_mz,
            bin_width=bin_width,
            top_n_peaks=top_n_peaks,
            min_separation_mz=1.5 * mz_tol,
        )
        target_mzs = _peaks.mzs.tolist()
        spectrum_df = pl.DataFrame(
            {
                "m_z": _peaks.mean_spectrum_mz,
                "intensity": _peaks.mean_spectrum_intensity,
            }
        )

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
            np.arange(1, n_passes + 1), dims="PASS_NUM", name="PASS_NUM"
        ),
    )
    passes_xr.name = "individual_peak_intensities"

    # ==========================================
    # MASK GENERATION
    # ==========================================

    pass1 = passes_xr.sel(PASS_NUM=1)
    mz_masks = compute_tissue_mask(pass1)

    # ==========================================================================

    masked_passes = passes_xr.where(mz_masks)

    stats_ds = compute_pass_statistics(masked_passes)
    return (
        masked_passes,
        spectrum_df,
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
    return


@app.cell
def _(spectrum_df, target_mzs):
    _mz_axis = spectrum_df["m_z"].to_numpy()
    _mean_spectrum = spectrum_df["intensity"].to_numpy()

    _peaks_df = pl.DataFrame(
        {
            "m_z": target_mzs,
            "intensity": [
                _mean_spectrum[
                    int(round((mz - _mz_axis[0]) / (_mz_axis[1] - _mz_axis[0])))
                ]
                for mz in target_mzs
            ],
        }
    )

    _spectrum_plot = spectrum_df.hvplot.line(
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
def _(stats_ds):
    value = "median"
    error = "std"

    norm_ds = xr.Dataset(
        {
            "value": stats_ds[value] / stats_ds[value].sel(PASS_NUM=1),
            "error": stats_ds[error] / stats_ds[value].sel(PASS_NUM=1),
        }
    )

    norm_ds = norm_ds.mean(dim="M_Z")

    show_export_button = True

    decay_df = pl.from_pandas(norm_ds.to_dataframe().reset_index())
    del norm_ds
    decay_df
    return decay_df, show_export_button


@app.cell
def _(decay_df):
    (
        decay_df.hvplot.errorbars(
            x="PASS_NUM", y="value", yerr1="error", width=400
        )
        * decay_df.hvplot.scatter(x="PASS_NUM", y="value", s=800)
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

    save_dataarray(
        passes_xr, path_list[0].parent / "multiple_passes_filtered.ome.zarr"
    )

    mo.md(f"""Saved decay profile as `{_out_path.name}` in:

    `{_out_path.parent}`

    """).callout(kind="success")
    return


@app.cell
def _(masked_passes, passes_xr):
    (
        passes_xr.hvplot.image(x="X", y="Y", aspect="square", cmap="magma")
        + masked_passes.hvplot.image(x="X", y="Y", aspect="square", cmap="magma")
    )
    return


@app.cell
def _(decay_df, n_passes):
    x_data = decay_df["PASS_NUM"].to_numpy()
    y_data = decay_df["value"].to_numpy()
    y_weights = decay_df["error"].to_numpy()

    fit_ds = fit_decay_models(x_data, y_data, y_weights, n_passes)
    return (fit_ds,)


@app.cell
def _():
    hvplot.output.info()
    return


@app.cell
def _(decay_df, fit_ds):
    error_bars = decay_df.hvplot.errorbars(
        x="PASS_NUM",
        y="value",
        yerr1="error",
        color="black",
        alpha=0.4,
        width=400,
    )

    scatter_plot = decay_df.hvplot.scatter(
        x="PASS_NUM",
        y="value",
        label="Observed (w/ Std Dev)",
        c="black",
        size=500,
        alpha=0.6,
        title="Weighted Decay Model Fitting",
    )


    fit_plots = fit_ds.drop_dims("PARAM_IDX").hvplot.line(
        x="X_FIT", y="y_fit", by="MODEL", alpha=0.4
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
