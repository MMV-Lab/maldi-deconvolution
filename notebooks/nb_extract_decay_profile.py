import marimo

__generated_with = "0.24.0"
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

    from bioio import BioImage
    from bioio_imzml import auto_pick_peaks, Reader

    from src.imzml_io import (
        read_imzml_to_xarray,
        bioimage_to_dataarray,
        save_dataarray,
    )
    from src.decay_profile import (
        compute_tissue_mask,
        compute_pass_statistics,
        fit_decay_models,
    )
    from src.plotting import hv_mpl_svg_render, relative_intensity_overlay

    hv.extension("matplotlib")
    hv.output(backend="matplotlib", fig="svg")

    hvplot.extension("matplotlib", "bokeh")
    hvplot.output(fig="svg")

    hvplot.output.info()
    hv.output.info()


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
        path_list,
        key=lambda x: int(re.search(r"(\d+)p(?:ass)?", x.name).group(1)),
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
    mz_tol = 0.005  # Da window around each peak for extraction
    top_n_peaks = 20  # How many automatic peaks to track
    bin_width = 0.001

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
            mz_tolerance_absolute=mz_tol,
            min_separation_absolute=mz_tol * 1.5,
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

    print(target_mzs)

    # Original
    # target_mzs = [760.586873524265, 798.5428735233675, 734.5708735248802, 772.5268735239827, 788.6178735236022, 761.5898735242413, 808.5718735231303, 799.5458735233437, 826.5738735227046, 735.5748735248565, 773.5308735239589, 762.6028735242173, 846.5278735222328, 789.6218735235784, 800.5588735233198, 782.5688735237452, 809.5748735231066, 666.4358735264914, 756.5528735243604, 836.6028735224675]

    # removed
    # target_mzs = [
    #     760.586873524265,
    #     798.5428735233675,
    #     734.5708735248802,
    #     772.5268735239827,
    #     788.6178735236022,
    #     826.5738735227046,
    #     846.5278735222328,
    #     782.5688735237452,
    #     666.4358735264914,
    #     756.5528735243604,
    #     836.6028735224675,
    # ]

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
        n_passes,
        passes_xr,
        spectrum_df,
        stats_ds,
        target_mzs,
    )


@app.cell
def _():
    # print("\n".join(f"{x:.4f}" for x in target_mzs))
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
                    int(
                        round((mz - _mz_axis[0]) / (_mz_axis[1] - _mz_axis[0]))
                    )
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
    hv_mpl_svg_render(_spectrum_plot * _peaks_plot)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Relative-intensity stability across oversampling passes

    How much does each tracked lipid's **relative** intensity vary from pass to
    pass? We look at two normalizations, both computed on tissue-masked pixels:

    * **Base-peak ratio** — each peak divided by the base peak (~760.585),
      matching the ratio used in `nb_figures`.
    * **Real TIC** — each peak divided by that pixel's *full-spectrum* total ion
      current (summed over the entire acquired m/z range, not just the tracked
      peaks), read straight from every imzML pass.

    For each lipid we take its per-pass mean relative intensity (`mean_rel`,
    averaged over tissue pixels), whose across-pass mean is `avg_rel`. We then
    quantify how much `mean_rel` moves from pass to pass, both in absolute terms
    and relative to that lipid's own `avg_rel`:

    * `std_rel` / `sem_rel` — plain std and standard error of `mean_rel` across
      passes (absolute units; `sem_rel = std_rel / √n_passes`)
    * `cv_pct`      = std_rel / avg_rel × 100  (coefficient of variation)
    * `max_dev_pct` = max |pass − avg_rel| / avg_rel × 100  (worst single pass)
    * `range_pct`   = (max − min) / avg_rel × 100  (full swing)

    The worst-case (max over lipids) `max_dev_pct` is the "did not vary more than
    XX%" figure; the median `cv_pct` is the typical value. The summary table
    below reports the full distribution of every metric across lipids, per
    normalization mode.

    *(Note: the `sem_rel` shown in the per-pass table / plot error bars is a
    different quantity — spread across pixels within one pass — whereas the
    `sem_rel` here is spread across passes.)*
    """)
    return


@app.cell
def _(n_passes, passes_xr, path_list):
    # Real TIC per pixel per pass: sum over the FULL binned spectrum of each
    # imzML (verified to reproduce the raw total ion current to <1e-6), not
    # just the tracked peaks. Reads via the same BioImage path as passes_xr, so
    # the Y/X grid lines up for a straight division.
    with mo.status.progress_bar(
        total=n_passes,
        title="Computing real TIC per pass",
        subtitle="Summing full spectra...",
    ) as _bar:
        _tic_arrays = []
        for _p in path_list:
            _full = bioimage_to_dataarray(
                BioImage(_p, reader=Reader), lazy=False
            )
            _tic_arrays.append(_full.sum(dim="M_Z"))  # (Y, X)
            _bar.update()

    tic_xr = xr.concat(
        _tic_arrays,
        dim=xr.DataArray(
            np.arange(1, n_passes + 1), dims="PASS_NUM", name="PASS_NUM"
        ),
    )
    tic_xr = tic_xr.assign_coords(Y=passes_xr.Y, X=passes_xr.X)
    tic_xr.name = "TIC"
    return (tic_xr,)


@app.cell
def _():
    # def _pass_tidy(rel: xr.DataArray, label: str) -> pl.DataFrame:
    #     """(PASS_NUM, M_Z, Y, X) relative intensity -> tidy per-pass mean/SEM.

    #     Each m/z carries its own tissue mask, so numerator and denominator
    #     masks don't perfectly overlap; drop the resulting inf/NaN pixels
    #     (reference/TIC absent) before averaging, mirroring the finite-filter
    #     in `nb_figures`. ``avg_rel`` is the across-pass mean, used to order
    #     lipids by abundance in the tables and plots.
    #     """
    #     rel = rel.where(np.isfinite(rel))
    #     summary = xr.Dataset(
    #         {
    #             "mean_rel": rel.mean(dim=["Y", "X"]),
    #             "sem_rel": rel.std(dim=["Y", "X"])
    #             / np.sqrt(rel.count(dim=["Y", "X"])),
    #         }
    #     )
    #     df = pl.from_pandas(summary.to_dataframe().reset_index()).rename(
    #         {"M_Z": "m/z", "PASS_NUM": "pass_num"}
    #     )
    #     avg = df.group_by("m/z").agg(
    #         pl.col("mean_rel").mean().alias("avg_rel")
    #     )
    #     return (
    #         df.join(avg, on="m/z")
    #         .with_columns(pl.lit(label).alias("normalization"))
    #         .sort(["avg_rel", "pass_num"], descending=[True, False])
    #     )

    # def _variation(summary: pl.DataFrame) -> pl.DataFrame:
    #     """Per-m/z spread of the per-pass mean, relative to each lipid's mean.

    #     ``std_rel``/``sem_rel`` are the plain (absolute) standard deviation and
    #     standard error of the mean of the per-pass relative intensity across
    #     passes; ``cv_pct`` is that std expressed as a percentage of ``avg_rel``.
    #     """
    #     return (
    #         summary.group_by("m/z", "normalization")
    #         .agg(
    #             pl.col("avg_rel").first(),
    #             pl.col("mean_rel").std().alias("std_rel"),
    #             pl.col("mean_rel").len().alias("_n"),
    #             pl.col("mean_rel").min().alias("_min"),
    #             pl.col("mean_rel").max().alias("_max"),
    #         )
    #         .with_columns(
    #             (pl.col("std_rel") / pl.col("_n").sqrt()).alias("sem_rel"),
    #             (pl.col("std_rel") / pl.col("avg_rel") * 100).alias("cv_pct"),
    #             (
    #                 pl.max_horizontal(
    #                     (pl.col("_max") - pl.col("avg_rel")).abs(),
    #                     (pl.col("avg_rel") - pl.col("_min")).abs(),
    #                 )
    #                 / pl.col("avg_rel")
    #                 * 100
    #             ).alias("max_dev_pct"),
    #             (
    #                 (pl.col("_max") - pl.col("_min")) / pl.col("avg_rel") * 100
    #             ).alias("range_pct"),
    #         )
    #         .drop("_n", "_min", "_max")
    #         .select(
    #             "m/z",
    #             "normalization",
    #             "avg_rel",
    #             "std_rel",
    #             "sem_rel",
    #             "cv_pct",
    #             "max_dev_pct",
    #             "range_pct",
    #         )
    #     )

    # # base-peak ratio vs ~760.585 (drop the trivially-constant reference)
    # _ref_mz = float(
    #     masked_passes.M_Z.sel(M_Z=760.5851, method="nearest").values
    # )
    # ratio_summary = _pass_tidy(
    #     masked_passes / masked_passes.sel(M_Z=_ref_mz),
    #     f"base-peak ({_ref_mz:.3f})",
    # ).filter(pl.col("m/z") != _ref_mz)

    # # real TIC: each peak / its pixel's full-spectrum total ion current
    # tic_summary = _pass_tidy(masked_passes / tic_xr, "real TIC")

    # variation_df = pl.concat(
    #     [_variation(ratio_summary), _variation(tic_summary)]
    # ).sort(["normalization", "avg_rel"], descending=[False, True])

    # # Full distribution of every metric across lipids, per normalization mode:
    # # count, min, quartiles, mean, median, max and std. `max` of `max_dev_pct`
    # # is the worst-case "did not vary more than XX%"; `median` of `cv_pct` is
    # # the typical variation.
    # _metrics = [
    #     "avg_rel",
    #     "std_rel",
    #     "sem_rel",
    #     "cv_pct",
    #     "max_dev_pct",
    #     "range_pct",
    # ]
    # _summary = (
    #     variation_df.unpivot(
    #         index=["normalization", "m/z"],
    #         on=_metrics,
    #         variable_name="metric",
    #         value_name="_v",
    #     )
    #     .group_by(["normalization", "metric"])
    #     .agg(
    #         pl.len().alias("n_lipids"),
    #         pl.col("_v").min().alias("min"),
    #         pl.col("_v").quantile(0.25).alias("q25"),
    #         pl.col("_v").median().alias("median"),
    #         pl.col("_v").mean().alias("mean"),
    #         pl.col("_v").quantile(0.75).alias("q75"),
    #         pl.col("_v").max().alias("max"),
    #         pl.col("_v").std().alias("std"),
    #     )
    #     .with_columns(
    #         pl.col("metric").cast(pl.Enum(_metrics)),  # keep logical order
    #         pl.col("min", "q25", "median", "mean", "q75", "max", "std").round(
    #             2
    #         ),
    #     )
    #     .sort(["normalization", "metric"])
    # )

    # mo.output.append(
    #     mo.md(
    #         "### Relative-intensity variation across passes "
    #         "(full stats per normalization)"
    #     )
    # )
    # mo.output.append(_summary)
    # mo.output.append(mo.md("Per-lipid detail:"))
    # mo.output.append(variation_df)
    return


@app.cell
def _(masked_passes, tic_xr):
    def _pass_tidy(rel: xr.DataArray, label: str) -> pl.DataFrame:
        """(PASS_NUM, M_Z, Y, X) relative intensity -> tidy per-pass mean/SEM.

        Each m/z carries its own tissue mask, so numerator and denominator
        masks don't perfectly overlap; drop the resulting inf/NaN pixels
        (reference/TIC absent) before averaging, mirroring the finite-filter
        in `nb_figures`. ``avg_rel`` is the across-pass mean, used to order
        lipids by abundance in the tables and plots.
        """
        rel = rel.where(np.isfinite(rel))
        summary = xr.Dataset(
            {
                "mean_rel": rel.mean(dim=["Y", "X"]),
                "sem_rel": rel.std(dim=["Y", "X"])
                / np.sqrt(rel.count(dim=["Y", "X"])),
            }
        )
        df = pl.from_pandas(summary.to_dataframe().reset_index()).rename(
            {"M_Z": "m/z", "PASS_NUM": "pass_num"}
        )
        avg = df.group_by("m/z").agg(
            pl.col("mean_rel").mean().alias("avg_rel")
        )
        return (
            df.join(avg, on="m/z")
            .with_columns(pl.lit(label).alias("normalization"))
            .sort(["avg_rel", "pass_num"], descending=[True, False])
        )

    def _variation(summary: pl.DataFrame) -> pl.DataFrame:
        """Per-m/z spread of the per-pass mean, relative to each lipid's mean.

        ``std_rel``/``sem_rel`` are the plain (absolute) standard deviation and
        standard error of the mean of the per-pass relative intensity across
        passes; ``cv_pct`` is that std expressed as a percentage of ``avg_rel``;
        ``dev_pct`` is the mean absolute percentage deviation across passes.
        """
        return (
            summary.group_by("m/z", "normalization")
            .agg(
                pl.col("avg_rel").first(),
                pl.col("mean_rel").std().alias("std_rel"),
                pl.col("mean_rel").len().alias("_n"),
                pl.col("mean_rel").min().alias("_min"),
                pl.col("mean_rel").max().alias("_max"),
                # Average absolute percentage deviation across passes
                (
                    (pl.col("mean_rel") - pl.col("avg_rel")).abs().mean()
                    / pl.col("avg_rel").first()
                    * 100
                ).alias("dev_pct"),
            )
            .with_columns(
                (pl.col("std_rel") / pl.col("_n").sqrt()).alias("sem_rel"),
                (pl.col("std_rel") / pl.col("avg_rel") * 100).alias("cv_pct"),
                (
                    pl.max_horizontal(
                        (pl.col("_max") - pl.col("avg_rel")).abs(),
                        (pl.col("avg_rel") - pl.col("_min")).abs(),
                    )
                    / pl.col("avg_rel")
                    * 100
                ).alias("max_dev_pct"),
                (
                    (pl.col("_max") - pl.col("_min")) / pl.col("avg_rel") * 100
                ).alias("range_pct"),
            )
            .drop("_n", "_min", "_max")
            .select(
                "m/z",
                "normalization",
                "avg_rel",
                "std_rel",
                "sem_rel",
                "cv_pct",
                "dev_pct",
                "max_dev_pct",
                "range_pct",
            )
        )


    # base-peak ratio vs ~760.585 (drop the trivially-constant reference)
    _ref_mz = float(masked_passes.M_Z.sel(M_Z=760.5851, method="nearest").values)
    ratio_summary = _pass_tidy(
        masked_passes / masked_passes.sel(M_Z=_ref_mz),
        f"base-peak ({_ref_mz:.3f})",
    ).filter(pl.col("m/z") != _ref_mz)

    # real TIC: each peak / its pixel's full-spectrum total ion current
    tic_summary = _pass_tidy(masked_passes / tic_xr, "real TIC")

    variation_df = pl.concat(
        [_variation(ratio_summary), _variation(tic_summary)]
    ).sort(["normalization", "avg_rel"], descending=[False, True])

    # Full distribution of every metric across lipids, per normalization mode:
    _metrics = [
        "avg_rel",
        "std_rel",
        "sem_rel",
        "cv_pct",
        "dev_pct",
        "max_dev_pct",
        "range_pct",
    ]
    _summary = (
        variation_df.unpivot(
            index=["normalization", "m/z"],
            on=_metrics,
            variable_name="metric",
            value_name="_v",
        )
        .group_by(["normalization", "metric"])
        .agg(
            pl.len().alias("n_lipids"),
            pl.col("_v").min().alias("min"),
            pl.col("_v").quantile(0.25).alias("q25"),
            pl.col("_v").median().alias("median"),
            pl.col("_v").mean().alias("mean"),
            pl.col("_v").quantile(0.75).alias("q75"),
            pl.col("_v").max().alias("max"),
            pl.col("_v").std().alias("std"),
        )
        .with_columns(
            pl.col("metric").cast(pl.Enum(_metrics)),  # keep logical order
            pl.col("min", "q25", "median", "mean", "q75", "max", "std").round(2),
        )
        .sort(["normalization", "metric"])
    )

    mo.output.append(
        mo.md(
            "### Relative-intensity variation across passes "
            "(full stats per normalization)"
        )
    )
    mo.output.append(_summary)
    mo.output.append(mo.md("Per-lipid detail:"))
    mo.output.append(variation_df)
    return ratio_summary, tic_summary


@app.cell
def _(ratio_summary):
    hv_mpl_svg_render(
        relative_intensity_overlay(
            ratio_summary, "Intensity ratio vs base peak"
        )
    )
    return


@app.cell
def _(tic_summary):
    hv_mpl_svg_render(
        relative_intensity_overlay(
            tic_summary, "Fraction of full-spectrum TIC"
        )
    )
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
        + masked_passes.hvplot.image(
            x="X", y="Y", aspect="square", cmap="magma"
        )
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
