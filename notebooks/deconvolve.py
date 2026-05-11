import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import polars as pl
    import hvplot
    import holoviews as hv
    import hvplot.xarray
    import numpy as np
    import xarray as xr

    import datetime
    import json

    from skimage.restoration import richardson_lucy
    from pathlib import Path
    from typing import Union, Tuple, Optional


    import sys

    _root_dir = Path(__file__).resolve().parent.parent
    if str(_root_dir) not in sys.path:
        sys.path.append(str(_root_dir))

    from src.psf_generator import MaldiPSFGenerator


@app.function
def load_decay_profile(
    source_input: Union[object, str, Path],
) -> Tuple[Optional[pl.DataFrame], Optional[str]]:
    """
    Parses a CSV file using Polars.
    Accepts 1 column (intensity) or 2 columns (pass, intensity).
    Handles files with or without headers.

    Args:
        source_input: Can be a marimo file object (from mo.ui.file()[0]),
                      a string path, or a pathlib.Path object.

    Returns:
        tuple: (pl.DataFrame | None, str | None) -> (Data, Error Message)
    """
    if not source_input:
        return None, "No file provided."

    # 1. NORMALIZE INPUT SOURCE
    # If it has .contents, it's a Marimo file object (bytes).
    # Otherwise, assume it's a Path/str.
    if hasattr(source_input, "contents"):
        source = source_input.contents  # Bytes
    elif isinstance(source_input, (str, Path)):
        source = source_input  # Path
    else:
        return (
            None,
            "Invalid input type. Must be a Marimo file object or a Path.",
        )

    try:
        # 2. READ STRATEGY: PEEK AT DATA
        # Read a few rows assuming no header to inspect the first row safely
        # Polars accepts both bytes and paths here.
        try:
            df_raw = pl.read_csv(source, has_header=False, n_rows=5)
        except Exception:
            # Captures empty files or non-CSV formats immediately
            return (
                None,
                "File cannot be read. It might be empty or not a valid CSV.",
            )

        if df_raw.is_empty():
            return None, "File is empty."

        # 3. DETECT HEADER
        # Check if the first row contains non-numeric strings
        first_row_vals = df_raw.row(0)
        has_header = False

        for val in first_row_vals:
            try:
                float(val)
            except (ValueError, TypeError):
                has_header = True
                break

        # 4. RELOAD WITH CORRECT SETTINGS
        # If we passed bytes, we can re-read the bytes variable.
        # If we passed a path, Polars just re-opens the file.
        df = pl.read_csv(source, has_header=has_header)

        # 5. STANDARDIZE COLUMNS
        width = df.width

        # CASE A: 1 Column (Intensity only)
        if width == 1:
            df = df.rename({df.columns[0]: "intensity"})
            # Generate a pass index (0, 1, 2...)
            df = df.with_columns(
                pl.int_range(0, pl.len(), dtype=pl.Int64).alias("pass_idx")
            )

        # CASE B: 2 Columns (Pass, Intensity)
        elif width == 2:
            # Assume col 0 is Pass, col 1 is Intensity
            df = df.select(
                [
                    pl.col(df.columns[0]).alias("pass_idx"),
                    pl.col(df.columns[1]).alias("intensity"),
                ]
            )

        # CASE C: 3+ Columns (Pass, Intensity, Error)
        elif width == 3:
            # Assume col 0 is Pass, col 1 is Intensity, col 2 is Error/Uncertainty
            df = df.select(
                [
                    pl.col(df.columns[0]).alias("pass_idx"),
                    pl.col(df.columns[1]).alias("intensity"),
                    pl.col(df.columns[2]).alias("error"),
                ]
            )

        else:
            return None, "CSV structure invalid."

        # 6. STRICT TYPE CASTING & CLEANUP
        # Drop nulls only on required columns; error column may have partial data
        df = df.drop_nulls(subset=["pass_idx", "intensity"])

        cast_exprs = [
            pl.col("pass_idx").cast(pl.Int64),
            pl.col("intensity").cast(pl.Float64),
        ]
        if "error" in df.columns:
            cast_exprs.append(pl.col("error").cast(pl.Float64))

        df = df.with_columns(cast_exprs)

        # Ensure pass_idx is sorted (optional, but usually desired for decay profiles)
        df = df.sort("pass_idx")

        # Reorder columns
        col_order = ["pass_idx", "intensity"]
        if "error" in df.columns:
            col_order.append("error")
        df = df.select(col_order)

        # Normalization logic depends on number of columns
        if "error" not in df.columns:
            # 1-2 column input: normalize to max
            max_intensity = df["intensity"].max()

            # Check if max is zero to avoid division by zero
            if max_intensity == 0:
                return None, "Maximum intensity is 0; cannot normalize data."

            # Normalize the intensity column
            df = df.with_columns(
                (pl.col("intensity") / max_intensity).alias("intensity")
            )
        else:
            # 3-column input: assume already normalized
            # Verify that the first intensity value is 1.0
            first_intensity = df["intensity"][0]
            tolerance = 1e-6
            if abs(first_intensity - 1.0) > tolerance:
                return (
                    None,
                    f"Validation Error: For 3-column input, intensity must be normalized with intensity[0]=1.0, got {first_intensity:.6f}.",
                )

        return df, None

    except FileNotFoundError:
        return None, f"File not found: {source}"
    except Exception as e:
        return None, f"Error processing file: {str(e)}"


@app.function
def validate_setup_params(values):
    """
    Validates the setup parameters form.
    Args:
        values (dict): A dictionary containing the current values of the inputs.
    Returns:
        str | None: An error message string if invalid, None if valid.
    """
    if not values:
        return "Form has no values."

    # 1. Validate Laser Diameter (Float > 0)
    try:
        laser = float(values["laser_diameter"])
        if laser <= 0:
            return "Laser Diameter must be greater than 0."
    except ValueError:
        return "Laser Diameter must be a valid number."

    # 2. Validate Step Size (Float > 0)
    try:
        step = float(values["step_size"])
        if step <= 0:
            return "Step Size must be greater than 0."
    except ValueError:
        return "Scanning Step Size must be a valid number."

    # 3. Validate Kernel Size (Int > 0, usually Odd)
    try:
        kernel = float(
            values["kernel_size"]
        )  # Allow float input but check if it's whole
        if not kernel.is_integer():
            return "Kernel Size must be a whole number (integer)."

        kernel = int(kernel)

        if kernel <= 0:
            return "Kernel Size must be a positive integer."

        # Optional: In image processing, kernels are usually odd numbers (3, 5, 7)
        if kernel % 2 == 0:
            return (
                "Kernel Size usually needs to be an odd number (e.g., 3, 5)."
            )

    except ValueError:
        return "Kernel Size must be a valid integer."

    # 4. Validate CSV Upload
    # mo.ui.file returns a list of files. Check if list is not empty.
    if not values["decay_csv"]:
        return "Please upload a Decay Profile CSV file."

    # print(values["decay_csv"][0])
    # print(type(values["decay_csv"][0]))

    # We actually run the loader here.
    # If it fails, the form shows the error.
    # If it succeeds, we discard the DF (we'll load it again later) and return None.
    _, csv_error = load_decay_profile(values["decay_csv"][0])

    if csv_error:
        return f"CSV Error: {csv_error}"

    # If all checks pass
    return None


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

    err = validate_setup_params(parameters_form.value)
    if isinstance(err, str):
        mo.output.append(mo.callout(str(err), kind="danger"))
    else:
        mo.output.append(mo.callout("Ok!", kind="success"))

        decay_df, _ = load_decay_profile(parameters_form.value["decay_csv"][0])
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
    return (maldi_psf_ds,)


@app.cell
def _(maldi_psf_ds):
    decay_plot = maldi_psf_ds.decay_measured.hvplot.scatter(
        x="pass_idx", label="Measured", color="black", size=50
    )

    if "decay_errors" in maldi_psf_ds:
        _err_data = {
            "pass_idx": maldi_psf_ds.pass_idx.values,
            "intensity": maldi_psf_ds.decay_measured.values,
            "error": maldi_psf_ds.decay_errors.values,
        }
        _err_bars = hv.ErrorBars(
            _err_data, kdims="pass_idx", vdims=["intensity", "error"]
        ).opts(color="black", line_width=1.5)
        decay_plot = decay_plot * _err_bars

    if "decay_fit" in maldi_psf_ds:
        fit_line = maldi_psf_ds.decay_fit.hvplot.line(
            x="pass_idx_fit", color="red", label="Fit", line_width=2
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

    hv.Layout(_plots).cols(2).opts("Image", invert_yaxis=True).opts(
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
            x="x_kernel",
            y="y_kernel",
            title=_title,
            cmap=_cmap,
            frame_width=250,
            frame_height=250,
            colorbar=True,
            hover=True,
            hover_tooltips=["val"],
        )

        _lbl = da.round(2).hvplot.labels(x="x_kernel", y="y_kernel", hover=False)

        _plots.append(_img * _lbl)


    hv.Layout(_plots).cols(3).opts(
        hv.opts.Overlay(invert_yaxis=True),
        hv.opts.Layout(title="Output Kernels (Coarse Grid)", shared_axes=False),
    )
    return


@app.cell
def _():
    widget_selected_dir = mo.ui.file_browser(
        multiple=False, selection_mode="file", filetypes=[".h5"]
    )
    mo.vstack(
        [
            mo.md("# Select the h5 file cointaing the original data:"),
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

        original_xr = xr.open_dataarray(input)
        original_xr = original_xr.chunk({"m_z": 1, "y": -1, "x": -1})
        _sim_pixel_size = float(parameters_form.value["step_size"])
        _image_pixel_size = float(original_xr.attrs.get("pixel_size_x"))
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


@app.function
def perform_batch_deconvolution(img_batch, kernel, num_iterations):
    """
    Operates on a chunk of data.
    Input shape:  (batch, y, x)  <- corresponds to a chunk of m_z
    Output shape: (batch, iteration, y, x)
    """
    # 1. Prepare output array
    # Shape: (batch_size, num_iterations + 1, y, x)
    batch_size, h, w = img_batch.shape
    out_shape = (batch_size, num_iterations + 1, h, w)
    output = np.zeros(out_shape, dtype=np.float32)

    # 2. Iterate over the batch (the m_z chunk)
    # We loop manually here, which is often faster/safer than np.vectorize
    for i in range(batch_size):
        img_2d = img_batch[i].astype(np.float32)

        # Normalize
        max_val = np.max(img_2d)
        if max_val > 0:
            img_2d /= max_val

        # Store Iteration 0 (Original)
        estimate = img_2d.copy()
        output[i, 0, :, :] = estimate

        # Deconvolution Loop
        for iter_idx in range(1, num_iterations + 1):
            estimate = richardson_lucy(estimate, kernel, num_iter=1)
            output[i, iter_idx, :, :] = estimate.astype(np.float32)

        # Rescale back to original intensity range
        output[i] *= max_val

    return output


@app.function
def deconvolute_dataset(input_xarray, kernel, num_iterations=10):

    iter_coords = np.arange(num_iterations + 1)

    deconvolved_da = xr.apply_ufunc(
        perform_batch_deconvolution,
        input_xarray,
        kwargs={"kernel": kernel, "num_iterations": num_iterations},
        input_core_dims=[["y", "x"]],
        output_core_dims=[["iteration", "y", "x"]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {"iteration": len(iter_coords)}},
    )

    deconvolved_da = deconvolved_da.assign_coords(iteration=iter_coords)

    deconvolved_da.name = "deconvolved_ion_intensity"
    deconvolved_da.attrs["long_name"] = "Ion Intensity"
    deconvolved_da.attrs["units"] = "a.u."
    return deconvolved_da


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
        x="x",
        y="y",
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
def _(maldi_psf_ds):
    maldi_psf_ds
    return


@app.cell
def _(export_button, input, maldi_psf_ds, result_xr):
    mo.stop(not export_button.value)

    with mo.status.spinner(
        title="Saving...", subtitle="This should take 1-2 minutes"
    ) as _spinner:
        out_dir = input.parent.parent / (
            "deconvolved_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        out_dir.mkdir(exist_ok=True, parents=True)

        maldi_psf_ds.attrs["decay_fit_results"] = json.dumps(
            maldi_psf_ds.attrs["decay_fit_results"]
        )
        maldi_psf_ds.to_netcdf(
            out_dir
            / f"{result_xr.attrs['original_file_name']}_simulation_data.h5",
        )

        result_xr.to_netcdf(
            out_dir / f"{result_xr.attrs['original_file_name']}_deconvolved.h5",
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
