import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import polars as pl
    import hvplot
    import holoviews as hv
    import hvplot.xarray
    import matplotlib.pyplot as plt
    import numpy as np

    import os
    import tifffile
    import itertools
    from skimage.restoration import richardson_lucy
    from pathlib import Path
    from typing import Union, Tuple, Optional

    from psf_generator import MaldiPSFGenerator

    # hvplot.extension("matplotlib")


@app.cell
def _():
    INPUT_DIR = "/mnt/eternus/users/Davide/maldi-deconv-rl/data/Sven_Chiahsin_Deconvolution_Oversampling_MALDI/Oversampled_U2OS-FUCCI-Cell/CC257_U2OS_1.3 µm"
    OUTPUT_DIR = "/mnt/eternus/users/Davide/maldi-deconv-rl/output/tmp"
    return


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
        elif width >= 2:
            # Assume col 0 is Pass, col 1 is Intensity
            df = df.select(
                [
                    pl.col(df.columns[0]).alias("pass_idx"),
                    pl.col(df.columns[1]).alias("intensity"),
                ]
            )

        else:
            return None, "CSV structure invalid."

        # 6. STRICT TYPE CASTING & CLEANUP
        # Drop nulls that might appear from empty trailing lines
        df = df.drop_nulls()

        df = df.with_columns(
            [
                pl.col("pass_idx").cast(pl.Int64),
                pl.col("intensity").cast(pl.Float64),
            ]
        )

        # Ensure pass_idx is sorted (optional, but usually desired for decay profiles)
        df = df.sort("pass_idx")

        # Reorder columns
        df = df.select(["pass_idx", "intensity"])

        max_intensity = df["intensity"].max()
        first_row_intensity = df["intensity"][0]

        # 1. Check if max is zero to avoid division by zero
        if max_intensity == 0:
            return None, "Maximum intensity is 0; cannot normalize data."

        # 2. Check if the maximum is in the first row
        # (We use a small tolerance or simple comparison)
        if first_row_intensity < max_intensity:
            return (
                None,
                "Validation Error: The maximum intensity must be in the first row (initial pass).",
            )

        # 3. Normalize the intensity column
        df = df.with_columns(
            (pl.col("intensity") / max_intensity).alias("intensity")
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
    parameters_form = (
        mo.md("""
        # Input the Setup Parameters

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
        maldi_psf = MaldiPSFGenerator(
            laser_diameter_um=float(parameters_form.value["laser_diameter"]),
            step_size_um=float(parameters_form.value["step_size"]),
            kernel_size=int(parameters_form.value["kernel_size"]),
            scan_direction=parameters_form.value["scan_dir"],
            decay_factors=decay_df["intensity"].to_list(),
            decay_model=parameters_form.value["decay_model"],
            method=parameters_form.value["sim_method"],
            sim_res_um=0.01,
        )
        ds = maldi_psf.to_xarray(fov_only=True)
    return ds, maldi_psf


@app.cell
def _(ds):
    decay_plot = ds.decay_measured.hvplot.scatter(
        x="pass_idx", label="Measured", color="black", size=50
    )

    if "decay_fit" in ds:
        fit_line = ds.decay_fit.hvplot.line(
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
def _():
    # import matplotlib_inline

    # matplotlib_inline.backend_inline.set_matplotlib_formats("svg")
    # plt.imshow(maldi_psf.decay_map, norm="linear")
    # plt.colorbar()
    # plt.show()
    return


@app.cell
def _(ds):
    _plot_configs = {
        "spatial_profile": ("1. Beam Spatial Profile", "Blues"),
        "dose_map": ("2. Accumulated Dose", "Oranges"),
        "decay_map": ("3. Decay Map (Damage)", "Viridis"),
        "final_signal": ("4. Final Emitted Signal", "Greens"),
    }

    _plots = []

    for _var, (_title, _cmap) in _plot_configs.items():
        _tmp = ds[_var].hvplot.image(
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
def _(maldi_psf):
    maldi_psf.step_size_um
    return


@app.cell
def _(ds):
    _plot_configs = {
        "spatial_kernel": ("A. Spatial Component", "Blues"),
        "pass_map": ("B. Exposure (Pass Map)", "Oranges"),
        "final_signal_kernel": ("C. Final PSF Kernel", "Greens"),
    }

    _plots = []

    for _var, (_title, _cmap) in _plot_configs.items():
        da = ds[_var].rename("val")

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
        multiple=False, selection_mode="directory"
    )
    mo.vstack(
        [
            mo.md("# Select the folder ofinput data"),
            widget_selected_dir,
        ]
    )
    return (widget_selected_dir,)


@app.cell
def _(widget_selected_dir):
    input_dir = None
    if len(widget_selected_dir.value) == 1:
        input_dir = Path(widget_selected_dir.value[0].path)
        output_dir = input_dir.parent / "deconvolved_tiffs"
        output_dir.mkdir(exist_ok=True, parents=True)
        # print(input_dir)
        # print(output_dir)
        mo.output.append(
            mo.md(
                f"""The initial data will be taken from this folder:

                `{input_dir}` 
            
                While the output data will be saved here:

                `{output_dir}` 
            
                """
            ).callout(kind="success")
        )
    return input_dir, output_dir


@app.cell
def _():
    # ds.final_signal_kernel.plot.imshow()
    return


@app.cell
def _(ds):
    ds.final_signal_kernel.values
    return


@app.cell
def _(ds, input_dir, maldi_psf, output_dir):
    mo.stop(not input_dir)

    file_patterns = ["*.tif", "*.tiff"]
    num_iterations = 10

    tif_files = []
    for pattern in file_patterns:
        tif_files.extend(input_dir.glob(pattern))

    for f in mo.status.progress_bar(tif_files):
        with tifffile.TiffFile(f) as tif:
            if not np.isclose(
                tif.pages[0].tags["XResolution"].value[0]
                / tif.pages[0].tags["XResolution"].value[1],
                1 / maldi_psf.step_size_um,
            ):
                break

        img = tifffile.imread(f)

        img = img.astype(np.float32)
        img = img / np.max(img) if np.max(img) > 0 else img

        iterations_stack = [img]
        estimate = img.copy()

        for iteration in range(1, num_iterations + 1):
            estimate = richardson_lucy(
                estimate, ds.final_signal_kernel.values, num_iter=1
            )
            iterations_stack.append(estimate.astype(np.float32))

        fname = f"{f.stem}_deconvolved.tiff"
        save_full_path = output_dir / fname

        tifffile.imwrite(
            save_full_path,
            np.array(iterations_stack),
            imagej=True,
            compression="zlib",
            photometric="minisblack",
            resolution=(1 / maldi_psf.step_size_um, 1 / maldi_psf.step_size_um),
            metadata={
                "axes": "TYX",
                "unit": "um",
            },
        )

    mo.output.append(
        mo.md(f"""
            Congratulations! 🎉
        
            All your `{len(tif_files)} files` are saved in:

            `{output_dir}`
    
        """).callout(kind="success")
    )
    return


if __name__ == "__main__":
    app.run()
