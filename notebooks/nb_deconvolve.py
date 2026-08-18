import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import hvplot
    import holoviews as hv
    import hvplot.xarray

    import datetime

    from pathlib import Path

    import sys

    _root_dir = Path(__file__).resolve().parent.parent
    if str(_root_dir) not in sys.path:
        sys.path.append(str(_root_dir))

    from src.psf_generator import MaldiPSFGenerator
    from src.deconvolution import (
        load_decay_profile,
        deconvolute_dataset,
    )
    from src.imzml_io import (
        DATAARRAY_EXTENSIONS,
        load_dataarray,
        pixel_sizes,
        save_dataarray,
    )
    from src.plotting import image_comparison_layout


@app.cell
def _():
    mo.md(r"""
    # Welcome!

    ## This notebook serves the purpose of simulating the beam profile and performing deconvolution.
    """)
    return


@app.cell
def _():
    parameters_form = (
        mo.md("""
        ## please insert the setup parameters below:

        | Parameter | Value |
        | ---: | :--- |
        | **Laser Diameter size** (measured at $1/e^2$): | `{laser_diameter}` $μm$ |
        | **Scanning Step Size:** | `{step_size}` $μm$ |
        | **Decay Profile** over multiple passes: | {decay_csv} *the file should contain either 2 columns (\#pass, average intensity) or 1 column (avg intensity). With or without headers can be accepted.*|
        | PSF Kernel Size: | `{kernel_size}` $px$ |
        | Scanning Direction: | `{scan_dir}` |
        | Simulation method: | `{sim_method}` |
        | Decay model: | `{decay_model}` |


    """)
        .batch(
            laser_diameter=mo.ui.text(value="4"),
            step_size=mo.ui.text(value="1.3"),
            kernel_size=mo.ui.text(value="3"),
            decay_csv=mo.ui.file(filetypes=[".csv"], kind="button"),
            scan_dir=mo.ui.dropdown(
                options={
                    "Top-Left → Bottom-Right": "tl_br",
                    "Bottom-Left → Top-Right": "bl_tr",
                },
                value="Top-Left → Bottom-Right",
            ),
            decay_model=mo.ui.dropdown(
                options={
                    "Exponential": "exponential",
                    "Linear Interpolation": "interpolation",
                },
                value="Exponential",
            ),
            sim_method=mo.ui.dropdown(
                options={
                    "Beam Profile Reconstruction": "beam_profile",
                    "Geometric Approximation": "geometric",
                },
                value="Beam Profile Reconstruction",
            ),
        )
        .form(show_clear_button=True)
    )

    parameters_form
    return (parameters_form,)


@app.cell
def _(parameters_form):
    mo.stop(not parameters_form.value)

    # No separate pre-validation: build straight from the form and let
    # MaldiPSFGenerator's own __post_init__ validation (and the CSV loader)
    # raise/report whatever is wrong.
    maldi_psf = None
    maldi_psf_ds = None
    try:
        if not parameters_form.value["decay_csv"]:
            raise ValueError("Please upload a Decay Profile CSV file.")

        decay_df, csv_error = load_decay_profile(
            parameters_form.value["decay_csv"][0]
        )
        if csv_error:
            raise ValueError(f"CSV Error: {csv_error}")

        decay_errors = (
            decay_df["error"].to_list() if "error" in decay_df.columns else None
        )
        maldi_psf = MaldiPSFGenerator(
            laser_diameter_um=float(parameters_form.value["laser_diameter"]),
            step_size_um=float(parameters_form.value["step_size"]),
            kernel_size=int(parameters_form.value["kernel_size"]),
            scan_direction=parameters_form.value["scan_dir"],
            decay_factors=decay_df["intensity"].to_list(),
            decay_errors=decay_errors,
            decay_model=parameters_form.value["decay_model"],
            method=parameters_form.value["sim_method"],
            sim_res_um=0.01,
        )
        maldi_psf_ds = maldi_psf.to_xarray(fov_only=False)
        mo.output.append(mo.callout("Ok!", kind="success"))
    except (ValueError, TypeError) as e:
        mo.output.append(mo.callout(str(e), kind="danger"))

    return maldi_psf, maldi_psf_ds


@app.cell
def _(maldi_psf_ds):
    decay_plot = maldi_psf_ds.decay_measured.hvplot.scatter(
        x="PASS_IDX", label="Measured", color="black", size=50
    )

    if "decay_errors" in maldi_psf_ds:
        _err_data = {
            "PASS_IDX": maldi_psf_ds.PASS_IDX.values,
            "intensity": maldi_psf_ds.decay_measured.values,
            "error": maldi_psf_ds.decay_errors.values,
        }
        _err_bars = hv.ErrorBars(
            _err_data, kdims="PASS_IDX", vdims=["intensity", "error"]
        ).opts(color="black", line_width=1.5)
        decay_plot = decay_plot * _err_bars

    if "decay_fit" in maldi_psf_ds:
        fit_line = maldi_psf_ds.decay_fit.hvplot.line(
            x="PASS_IDX_FIT", color="red", label="Fit", line_width=2
        )
        decay_plot = decay_plot * fit_line

    decay_plot.opts(
        title="Signal Decay Profile",
        width=600,
        height=400,
        legend_position="top_right",
        show_grid=True,
    )
    return


@app.cell
def _(maldi_psf_ds):
    _plot_configs = {
        "spatial_profile": ("1. Beam Spatial Profile", "Blues"),
        "dose_map": ("2. Accumulated Dose", "Oranges"),
        "decay_map": ("3. Decay Map (Damage)", "Viridis"),
        "final_signal": ("4. Final Emitted Signal", "Greens"),
    }

    _plots = []

    for _var, (_title, _cmap) in _plot_configs.items():
        _tmp = maldi_psf_ds[_var].hvplot.image(
            title=_title,
            cmap=_cmap,
            aspect="square",
            data_aspect=1,
            # frame_width=250,
            frame_height=250,
            rasterize=True,
            hover=True,
            hover_tooltips=[_var],
            toolbar=None,
        )
        _plots.append(_tmp)

    image_comparison_layout(
        _plots,
        cols=2,
        opts_group="Image",
        invert_yaxis=True,
        shared_axes=True,
        title="Simulation Maps (Fine Grid)",
    )
    return


@app.cell
def _(maldi_psf_ds):
    _plot_configs = {
        "spatial_kernel": ("A. Spatial Component", "Blues"),
        "pass_map": ("B. Exposure (Pass Map)", "Oranges"),
        "final_signal_kernel": ("C. Final PSF Kernel", "Greens"),
    }

    _plots = []

    for _var, (_title, _cmap) in _plot_configs.items():
        da = maldi_psf_ds[_var].rename("val")

        _img = da.hvplot.image(
            x="X_KERNEL",
            y="Y_KERNEL",
            title=_title,
            cmap=_cmap,
            frame_width=250,
            frame_height=250,
            colorbar=True,
            hover=True,
            hover_tooltips=["val"],
        )

        _lbl = da.round(2).hvplot.labels(x="X_KERNEL", y="Y_KERNEL", hover=False)

        _plots.append(_img * _lbl)


    image_comparison_layout(
        _plots,
        cols=3,
        opts_group="Overlay",
        invert_yaxis=True,
        shared_axes=False,
        title="Output Kernels (Coarse Grid)",
    )
    return


@app.cell
def _():
    widget_selected_dir = mo.ui.file_browser(
        multiple=False,
        selection_mode="all",
        filetypes=DATAARRAY_EXTENSIONS,
        initial_path="data",
    )
    mo.vstack(
        [
            mo.md("# Select the original data (`.ome.zarr` store):"),
            widget_selected_dir,
        ]
    )
    return (widget_selected_dir,)


@app.cell
def _(parameters_form, widget_selected_dir):
    input = None
    if len(widget_selected_dir.value) == 1:
        input = Path(widget_selected_dir.value[0].path)
        mo.output.append(
            mo.md(
                "Found!"
                # f"""The initial data will be taken from this folder:
                # `{input_dir}`
                # While the output data will be saved here:
                # `{out_dir}`
                # """
            ).callout(kind="success")
        )

        original_xr = load_dataarray(input)
        original_xr = original_xr.chunk({"M_Z": 1, "Y": -1, "X": -1})
        _sim_pixel_size = float(parameters_form.value["step_size"])
        _image_pixel_size = pixel_sizes(original_xr)[1]
        if _image_pixel_size != _sim_pixel_size:
            # del original_xr
            mo.output.replace(
                mo.md(
                    f"""The pixel size do *not* match!

                    **Image**: {_image_pixel_size}

                    **Simulation**: {_sim_pixel_size}

                    """
                ).callout(kind="danger")
            )
    return input, original_xr


@app.cell
def _(maldi_psf_ds, original_xr):
    result_xr = deconvolute_dataset(
        original_xr,
        kernel=maldi_psf_ds.final_signal_kernel.values,
        num_iterations=25,
    )
    return (result_xr,)


@app.cell
def _(result_xr):
    _preview = result_xr.hvplot.image(
        x="X",
        y="Y",
        data_aspect=1,
        cmap="magma",
        flip_yaxis=True,
        responsive=True,
        min_height=300,
        widget_location="top_left",
    )

    mo.output.append(mo.md("## Preview"))
    mo.output.append(_preview)
    return


@app.cell
def _(result_xr):
    result_xr

    export_button = mo.ui.run_button(label="Export")

    mo.vstack(
        [
            mo.md(
                "### Press the following button to save the deconvolved results"
            ),
            export_button,
        ]
    )
    return (export_button,)


@app.cell
def _(export_button, input, maldi_psf, result_xr):
    mo.stop(not export_button.value)

    with mo.status.spinner(
        title="Saving...", subtitle="This should take 1-2 minutes"
    ) as _spinner:
        out_dir = input.parent.parent / (
            "deconvolved_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        out_dir.mkdir(exist_ok=True, parents=True)

        # Base output name: the input store's name minus the suffixes the
        # extract notebook appended (e.g. "X.imzml_original.ome.zarr" -> "X.imzml").
        _base_name = input.name.removesuffix(".ome.zarr").removesuffix("_original")

        # Only the (tiny) constructor params + coarse kernels are persisted
        # -- MaldiPSFGenerator.save (see its docstring) skips the fine-grid
        # simulation arrays entirely since they're deterministically
        # re-derivable from these params alone.
        maldi_psf.save(
            out_dir / f"{_base_name}_simulation_data.json",
        )

        save_dataarray(
            result_xr,
            out_dir / f"{_base_name}_deconvolved.ome.zarr",
        )

    mo.output.append(
        mo.md(f"""
            Congratulations! 🎉

            All your files are saved in:

            `{out_dir}`

        """).callout(kind="success")
    )
    return


if __name__ == "__main__":
    app.run()
