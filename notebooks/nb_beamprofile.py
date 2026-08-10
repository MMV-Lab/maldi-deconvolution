import marimo

__generated_with = "0.23.14"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import math
    import numpy as np
    import hvplot
    import xarray as xr

    import polars as pl

    import holoviews as hv
    from holoviews import opts

    hv.extension("bokeh")

    from astropy.modeling import models, fitting


    # import hvplot.xarray
    # import hvplot.polars
    # import hvplot.pandas


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Welcome!

    ## This notebook serves the purpose of extracting the beam profile
    """)
    return


@app.cell
def _():
    widget_selected_file = mo.ui.file_browser(filetypes=[".cma"], multiple=True)
    mo.vstack([mo.md("# Select `cma` file"), widget_selected_file])
    return (widget_selected_file,)


@app.cell
def _():
    return


@app.cell
def _(widget_selected_file):
    path_list = [i.path for i in widget_selected_file.value]
    path_list = sorted(path_list)
    return (path_list,)


@app.cell
def _():
    x_pixel_size = 0.230  # um
    y_pixel_size = 0.540  # um
    return x_pixel_size, y_pixel_size


@app.cell
def _(path_list):
    # beware of trailing comma
    images = [
        pl.read_csv(p, has_header=False).to_numpy()[:, :-1].astype(float)
        for p in path_list
    ]
    return (images,)


@app.cell
def _(images, x_pixel_size, y_pixel_size):
    _stacked_data = np.stack(images)

    xy_profiles = xr.DataArray(
        _stacked_data,
        dims=["sample", "y", "x"],
        coords={
            "sample": np.arange(_stacked_data.shape[0]),
            "y": np.arange(_stacked_data.shape[1]) * y_pixel_size,
            "x": np.arange(_stacked_data.shape[2]) * x_pixel_size,
        },
        name="intensity",
        attrs={"units": "um"},
    )

    max_pos = xy_profiles.mean(dim="sample").argmax(dim=["y", "x"])

    N = 20
    y_mid = int(max_pos["y"])
    x_mid = int(max_pos["x"])

    xy_profiles = xy_profiles.isel(
        y=slice(y_mid - N // 2, y_mid + N // 2),
        x=slice(x_mid - N // 2, x_mid + N // 2),
    )
    return (xy_profiles,)


@app.cell
def _(xy_profiles):
    x_coords = xy_profiles.x.values
    y_coords = xy_profiles.y.values
    X, Y = np.meshgrid(x_coords, y_coords)

    avg_img = xy_profiles.mean(dim="sample")
    std_img = xy_profiles.std(dim="sample")

    fitter = fitting.LevMarLSQFitter()
    return X, Y, avg_img, fitter, std_img, x_coords, y_coords


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell(column=1)
def _(X, Y, avg_img, fitter, x_coords, y_coords):
    _z_data = avg_img.values


    _max_idx = np.unravel_index(np.argmax(_z_data), _z_data.shape)
    _y_peak_guess = y_coords[_max_idx[0]]
    _x_peak_guess = x_coords[_max_idx[1]]

    _model_init = models.Gaussian2D(
        amplitude=_z_data.max(),
        x_mean=_x_peak_guess,
        y_mean=_y_peak_guess,
        x_stddev=1.0,
        y_stddev=1.0,
    )

    global_best_fit = fitter(_model_init, X, Y, _z_data)


    # stds = std_img.values
    # positive_stds = stds[stds > 0]

    # if len(positive_stds) > 0:
    #     noise_floor = np.median(positive_stds)
    # else:
    #     noise_floor = 1e-6

    # safe_std = np.clip(stds, a_min=noise_floor, a_max=None)

    # weights = 1.0 / safe_std
    # best_fit = fitter(model_init, X, Y, z_data, weights=weights, filter_non_finite=True)
    return (global_best_fit,)


@app.cell
def _(X, Y, avg_img, global_best_fit, std_img):
    hv.extension("matplotlib")

    # Define the dimensions with units
    x_dim = hv.Dimension("x", label=r"x ($\mu$m)")
    y_dim = hv.Dimension("y", label=r"y ($\mu$m)")

    x_center = global_best_fit.x_mean.value
    y_center = global_best_fit.y_mean.value

    idx_x = np.abs(avg_img.x.values - x_center).argmin()
    idx_y = np.abs(avg_img.y.values - y_center).argmin()

    # Prepare X-profile data (horizontal slice at y_center)
    x_vals = avg_img.x.values
    x_data_prof = avg_img.values[idx_y, :]
    x_err_prof = std_img.values[idx_y, :]
    x_fit_prof = global_best_fit(x_vals, y_center)

    # Prepare Y-profile data (vertical slice at x_center)
    y_vals = avg_img.y.values
    y_data_prof = avg_img.values[:, idx_x]
    y_err_prof = std_img.values[:, idx_x]
    y_fit_prof = global_best_fit(x_center, y_vals)

    # Main Plot: Original Image + Fit Contours
    img = hv.Image(
        (x_vals, y_vals, avg_img.values), kdims=[x_dim, y_dim], vdims=["Intensity"]
    )

    fit_img = hv.Image(
        (x_vals, y_vals, global_best_fit(X, Y)),
        kdims=[x_dim, y_dim],
        vdims=["Intensity"],
    )
    contours = hv.operation.contours(fit_img, levels=6)
    main_plot = img * contours

    # Top Pane: X-profile (Data + Errors + Fit)
    top_scatter = hv.Scatter(
        (x_vals, x_data_prof), kdims=[x_dim], vdims=["Intensity"]
    )
    top_errs = hv.ErrorBars(
        (x_vals, x_data_prof, x_err_prof),
        kdims=[x_dim],
        vdims=["Intensity", "Error"],
    )
    top_fit = hv.Curve((x_vals, x_fit_prof), kdims=[x_dim], vdims=["Intensity"])

    top_plot = top_scatter * top_errs * top_fit

    # Right Pane: Y-profile (Data + Errors + Fit)
    right_scatter = hv.Scatter(
        (y_vals, y_data_prof), kdims=[y_dim], vdims=["Intensity"]
    )
    right_errs = hv.ErrorBars(
        (y_vals, y_data_prof, y_err_prof),
        kdims=[y_dim],
        vdims=["Intensity", "Error"],
    )
    right_fit = hv.Curve((y_vals, y_fit_prof), kdims=[y_dim], vdims=["Intensity"])

    right_plot = right_scatter * right_errs * right_fit

    layout = main_plot << right_plot << top_plot

    layout.opts(
        opts.Image(cmap="viridis"),
        opts.Contours(color="white", linewidth=1.5),
        opts.Scatter(color="black", marker="o"),
        opts.ErrorBars(color="black", linewidth=1),
        opts.Curve(color="red", linewidth=2),
        opts.AdjointLayout(),  # Scales the overall figure size
    )
    return (layout,)


@app.cell
def _(layout):
    hv.render(layout)
    return


@app.cell
def _(layout, path_list):
    hv.save(layout, f"{path_list[0].parent.parent}.svg", fmt="svg", dpi=1200)
    return


@app.cell
def _(global_best_fit):
    sig_x = global_best_fit.x_stddev.value
    sig_y = global_best_fit.y_stddev.value

    print(f"X-direction 1/e^2 diameter: {4 * sig_x:.4f} um")
    print(f"Y-direction 1/e^2 diameter: {4 * sig_y:.4f} um ")
    print(f"Beam Ellipticity : {min(sig_x, sig_y) / max(sig_x, sig_y):.4f} ")

    print(f"Geometric Mean 1/e^2 diameter: {4 * np.sqrt(sig_x * sig_y):.4f} um")
    print(f"Aritmetic Mean 1/e^2 diameter: {2 * (sig_x + sig_y):.4f} um")
    return


@app.cell
def _(X, Y, avg_img, global_best_fit, std_img):
    hv.extension("matplotlib")

    # Define the dimensions with units
    _x_dim = hv.Dimension("x", label=r"x ($\mu$m)")
    _y_dim = hv.Dimension("y", label=r"y ($\mu$m)")

    # Peak location (the "maximum") in the original coordinate system
    _x_center = global_best_fit.x_mean.value
    _y_center = global_best_fit.y_mean.value

    # Original coordinate arrays (needed for indexing / evaluating the fit)
    _x_vals_orig = avg_img.x.values
    _y_vals_orig = avg_img.y.values

    # Index of the slice through the peak
    _idx_x = np.abs(_x_vals_orig - _x_center).argmin()
    _idx_y = np.abs(_y_vals_orig - _y_center).argmin()

    # Shifted coordinates so the peak sits at (0, 0)
    _x_vals = _x_vals_orig - _x_center
    _y_vals = _y_vals_orig - _y_center

    # Prepare X-profile data (horizontal slice at y_center)
    _x_data_prof = avg_img.values[_idx_y, :]
    _x_err_prof = std_img.values[_idx_y, :]
    _x_fit_prof = global_best_fit(_x_vals_orig, _y_center)

    # Prepare Y-profile data (vertical slice at x_center)
    _y_data_prof = avg_img.values[:, _idx_x]
    _y_err_prof = std_img.values[:, _idx_x]
    _y_fit_prof = global_best_fit(_x_center, _y_vals_orig)

    # Main Plot: Original Image + Fit Contours (shown in peak-centered coordinates)
    _img = hv.Image(
        (_x_vals, _y_vals, avg_img.values), kdims=[_x_dim, _y_dim], vdims=["Intensity"]
    )
    _fit_img = hv.Image(
        (_x_vals, _y_vals, global_best_fit(X, Y)),
        kdims=[_x_dim, _y_dim],
        vdims=["Intensity"],
    )
    _contours = hv.operation.contours(_fit_img, levels=6)
    _main_plot = _img * _contours

    # Top Pane: X-profile (Data + Errors + Fit)
    _top_scatter = hv.Scatter(
        (_x_vals, _x_data_prof), kdims=[_x_dim], vdims=["Intensity"]
    )
    _top_errs = hv.ErrorBars(
        (_x_vals, _x_data_prof, _x_err_prof),
        kdims=[_x_dim],
        vdims=["Intensity", "Error"],
    )
    _top_fit = hv.Curve((_x_vals, _x_fit_prof), kdims=[_x_dim], vdims=["Intensity"])
    _top_plot = _top_scatter * _top_errs * _top_fit

    # Right Pane: Y-profile (Data + Errors + Fit)
    _right_scatter = hv.Scatter(
        (_y_vals, _y_data_prof), kdims=[_y_dim], vdims=["Intensity"]
    )
    _right_errs = hv.ErrorBars(
        (_y_vals, _y_data_prof, _y_err_prof),
        kdims=[_y_dim],
        vdims=["Intensity", "Error"],
    )
    _right_fit = hv.Curve((_y_vals, _y_fit_prof), kdims=[_y_dim], vdims=["Intensity"])
    _right_plot = _right_scatter * _right_errs * _right_fit

    _layout = _main_plot << _right_plot << _top_plot
    _layout.opts(
        opts.Image(cmap="viridis"),
        opts.Contours(color="white", linewidth=1.5),
        opts.Scatter(color="black", marker="o"),
        opts.ErrorBars(color="black", linewidth=1),
        opts.Curve(color="red", linewidth=2),
        opts.AdjointLayout(),  # Scales the overall figure size
    )

    hv.render(_layout)
    hv.save(_layout, "newbeamprofile.svg", fmt="svg", dpi=1200)
    return


@app.cell(column=2)
def _(X, Y, fitter, x_coords, xy_profiles, y_coords):
    fit_results = []

    for _sample_val in xy_profiles.sample.values:
        _z_data = xy_profiles.sel(sample=_sample_val).values

        _max_idx = np.unravel_index(np.argmax(_z_data), _z_data.shape)
        _y_peak_guess = y_coords[_max_idx[0]]
        _x_peak_guess = x_coords[_max_idx[1]]

        _model_init = models.Gaussian2D(
            amplitude=_z_data.max(),
            x_mean=_x_peak_guess,
            y_mean=_y_peak_guess,
            x_stddev=1.0,
            y_stddev=1.0,
        )

        _fit = fitter(_model_init, X, Y, _z_data)

        fit_results.append(
            {
                "sample": _sample_val,
                "amplitude": _fit.amplitude.value,
                "x_mean": _fit.x_mean.value,
                "y_mean": _fit.y_mean.value,
                "x_stddev": _fit.x_stddev.value,
                "y_stddev": _fit.y_stddev.value,
            }
        )


    # mean_params = df_sample_fits.select(pl.exclude("sample").mean()).row(0, named=True)
    # sample_best_fit = models.Gaussian2D(**mean_params)
    return (fit_results,)


@app.cell
def _(fit_results):
    df_sample_fits = pl.DataFrame(fit_results)

    df_sample_fits = (
        pl.DataFrame(fit_results)
        .with_columns(sigma_eff=(pl.col("x_stddev") * pl.col("y_stddev")).sqrt())
        .with_columns(beam_diameter_1e2=pl.col("sigma_eff") * 4)
    )

    df_sample_fits
    return (df_sample_fits,)


@app.cell
def _(df_sample_fits):
    print(
        f"Value ± std = {df_sample_fits['beam_diameter_1e2'].mean():.04f} ± {df_sample_fits['beam_diameter_1e2'].std():.04f} um "
    )
    print(
        f"Value ± sem = {df_sample_fits['beam_diameter_1e2'].mean():.04f} ± {df_sample_fits['beam_diameter_1e2'].std() / math.sqrt(df_sample_fits['beam_diameter_1e2'].count()):.04f} um "
    )
    return


if __name__ == "__main__":
    app.run()
