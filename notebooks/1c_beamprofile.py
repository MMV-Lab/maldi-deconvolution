import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import hvplot
    import xarray as xr

    import polars as pl

    import holoviews as hv
    from holoviews import opts

    hv.extension("bokeh")


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
def _(widget_selected_file):
    path_list = [i.path for i in widget_selected_file.value]
    path_list = sorted(path_list)

    # [i.name for i in path_list]
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
    return (xy_profiles,)


@app.cell
def _(xy_profiles):
    max_pos = xy_profiles.mean(dim="sample").argmax(dim=["y", "x"])

    N = 20
    half = N // 2
    y_mid = int(max_pos["y"])
    x_mid = int(max_pos["x"])

    cropped = xy_profiles.isel(
        y=slice(y_mid - half, y_mid + half), x=slice(x_mid - half, x_mid + half)
    )
    return (cropped,)


@app.cell
def _(cropped):
    from astropy.modeling import models, fitting

    avg_img = cropped.mean(dim="sample")
    std_img = cropped.std(dim="sample")

    z_data = avg_img.values
    x_coords = avg_img.x.values
    y_coords = avg_img.y.values
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

    stds = std_img.values
    positive_stds = stds[stds > 0]

    if len(positive_stds) > 0:
        noise_floor = np.median(positive_stds)
    else:
        noise_floor = 1e-6

    safe_std = np.clip(stds, a_min=noise_floor, a_max=None)

    weights = 1.0 / safe_std

    # best_fit = fitter(model_init, X, Y, z_data, weights=weights, filter_non_finite=True)
    best_fit = fitter(model_init, X, Y, z_data)
    return X, Y, avg_img, best_fit, std_img


@app.cell
def _(X, Y, avg_img, best_fit, std_img):
    # Activate matplotlib backend
    hv.extension("matplotlib")

    # Get the center coordinates from the fit
    x_center = best_fit.x_mean.value
    y_center = best_fit.y_mean.value

    idx_x = np.abs(avg_img.x.values - x_center).argmin()
    idx_y = np.abs(avg_img.y.values - y_center).argmin()

    # Prepare X-profile data (horizontal slice at y_center)
    x_vals = avg_img.x.values
    x_data_prof = avg_img.values[idx_y, :]
    x_err_prof = std_img.values[idx_y, :]
    x_fit_prof = best_fit(x_vals, y_center)

    # Prepare Y-profile data (vertical slice at x_center)
    y_vals = avg_img.y.values
    y_data_prof = avg_img.values[:, idx_x]
    y_err_prof = std_img.values[:, idx_x]
    y_fit_prof = best_fit(x_center, y_vals)

    # Main Plot: Original Image + Fit Contours
    img = hv.Image(
        (x_vals, y_vals, avg_img.values), kdims=["x", "y"], vdims=["Intensity"]
    )

    fit_img = hv.Image(
        (x_vals, y_vals, best_fit(X, Y)), kdims=["x", "y"], vdims=["Intensity"]
    )
    contours = hv.operation.contours(fit_img, levels=6)
    main_plot = img * contours

    # Top Pane: X-profile (Data + Errors + Fit)
    top_scatter = hv.Scatter(
        (x_vals, x_data_prof), kdims=["x"], vdims=["Intensity"]
    )
    top_errs = hv.ErrorBars(
        (x_vals, x_data_prof, x_err_prof),
        kdims=["x"],
        vdims=["Intensity", "Error"],
    )
    top_fit = hv.Curve((x_vals, x_fit_prof), kdims=["x"], vdims=["Intensity"])

    top_plot = top_scatter * top_errs * top_fit

    # Right Pane: Y-profile (Data + Errors + Fit)
    right_scatter = hv.Scatter(
        (y_vals, y_data_prof), kdims=["y"], vdims=["Intensity"]
    )
    right_errs = hv.ErrorBars(
        (y_vals, y_data_prof, y_err_prof),
        kdims=["y"],
        vdims=["Intensity", "Error"],
    )
    right_fit = hv.Curve((y_vals, y_fit_prof), kdims=["y"], vdims=["Intensity"])

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


@app.cell(hide_code=True)
def _(best_fit):
    sig_x = best_fit.x_stddev.value
    sig_y = best_fit.y_stddev.value

    print(f"X-direction 1/e^2 diameter: {4 * sig_x:.4f} um")
    print(f"Y-direction 1/e^2 diameter: {4 * sig_y:.4f} um ")
    print(f"Beam Ellipticity : {min(sig_x,sig_y)/max(sig_x,sig_y):.4f} ")

    print(f"Geometric Mean 1/e^2 diameter: {4* np.sqrt(sig_x*sig_y):.4f} um")
    print(f"Aritmetic Mean 1/e^2 diameter: {2 * (sig_x+sig_y):.4f} um")
    return


if __name__ == "__main__":
    app.run()
