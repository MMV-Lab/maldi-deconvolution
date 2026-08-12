import marimo

__generated_with = "0.23.14"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import numpy as np

    import holoviews as hv
    from holoviews import opts

    import sys
    from pathlib import Path as _Path

    _root_dir = _Path(__file__).resolve().parent.parent
    if str(_root_dir) not in sys.path:
        sys.path.append(str(_root_dir))

    from src.beam_profile import (
        load_cma_images,
        crop_around_peak,
        fit_gaussian2d,
        beam_diameter_stats,
    )
    from src.plotting import set_hv_backend

    set_hv_backend("bokeh", fig_format=None)

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
def _(path_list, x_pixel_size, y_pixel_size):
    xy_profiles_full = load_cma_images(path_list, x_pixel_size, y_pixel_size)
    return (xy_profiles_full,)


@app.cell
def _(xy_profiles_full):
    xy_profiles = crop_around_peak(xy_profiles_full, window=20)
    return (xy_profiles,)


@app.cell
def _(xy_profiles):
    x_coords = xy_profiles.X.values
    y_coords = xy_profiles.Y.values
    X, Y = np.meshgrid(x_coords, y_coords)

    avg_img = xy_profiles.mean(dim="SAMPLE")
    std_img = xy_profiles.std(dim="SAMPLE")
    return X, Y, avg_img, std_img, x_coords, y_coords


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
def _(avg_img, x_coords, y_coords):
    global_best_fit = fit_gaussian2d(x_coords, y_coords, avg_img.values)

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
    set_hv_backend("matplotlib", fig_format=None)

    # Define the dimensions with units
    x_dim = hv.Dimension("x", label=r"x ($\mu$m)")
    y_dim = hv.Dimension("y", label=r"y ($\mu$m)")

    x_center = global_best_fit.x_mean.value
    y_center = global_best_fit.y_mean.value

    idx_x = np.abs(avg_img.X.values - x_center).argmin()
    idx_y = np.abs(avg_img.Y.values - y_center).argmin()

    # Prepare X-profile data (horizontal slice at y_center)
    x_vals = avg_img.X.values
    x_data_prof = avg_img.values[idx_y, :]
    x_err_prof = std_img.values[idx_y, :]
    x_fit_prof = global_best_fit(x_vals, y_center)

    # Prepare Y-profile data (vertical slice at x_center)
    y_vals = avg_img.Y.values
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
    set_hv_backend("matplotlib", fig_format=None)

    # Define the dimensions with units
    _x_dim = hv.Dimension("x", label=r"x ($\mu$m)")
    _y_dim = hv.Dimension("y", label=r"y ($\mu$m)")

    # Peak location (the "maximum") in the original coordinate system
    _x_center = global_best_fit.x_mean.value
    _y_center = global_best_fit.y_mean.value

    # Original coordinate arrays (needed for indexing / evaluating the fit)
    _x_vals_orig = avg_img.X.values
    _y_vals_orig = avg_img.Y.values

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
def _(x_coords, xy_profiles, y_coords):
    fit_results = []

    for _sample_val in xy_profiles.SAMPLE.values:
        _fit = fit_gaussian2d(
            x_coords, y_coords, xy_profiles.sel(SAMPLE=_sample_val).values
        )

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
    df_sample_fits, beam_diameter_summary = beam_diameter_stats(fit_results)

    df_sample_fits
    return (df_sample_fits, beam_diameter_summary)


@app.cell
def _(beam_diameter_summary):
    print(
        f"Value ± std = {beam_diameter_summary['mean']:.04f} ± {beam_diameter_summary['std']:.04f} um "
    )
    print(
        f"Value ± sem = {beam_diameter_summary['mean']:.04f} ± {beam_diameter_summary['sem']:.04f} um "
    )
    return


if __name__ == "__main__":
    app.run()
