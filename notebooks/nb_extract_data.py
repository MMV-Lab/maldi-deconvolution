import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    from pathlib import Path
    import hvplot
    import hvplot.xarray
    import datetime

    import sys

    _root_dir = Path(__file__).resolve().parent.parent
    if str(_root_dir) not in sys.path:
        sys.path.append(str(_root_dir))

    from src.imzml_io import (
        read_imzml_to_xarray,
        parse_mz_csv,
        pixel_sizes,
        save_dataarray,
    )

    from bioio_imzml import auto_pick_peaks


@app.cell
def _():
    mo.md(r"""
    # Welcome!

    ## This notebook serves the purpose of extracting data from a `imzML` file to an OME-Zarr (`.ome.zarr`) file which can be processed later (deconvolution or visualization)
    """)
    return


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _():
    widget_selected_file = mo.ui.file_browser(
        filetypes=[".imzml"], multiple=False, restrict_navigation=False
    )
    mo.vstack([mo.md("# Select `imzml` file"), widget_selected_file])
    return (widget_selected_file,)


@app.cell
def _(is_script_mode, widget_selected_file):
    filepath = None

    if is_script_mode:
        _raw = mo.cli_args().get("input")
        if _raw:
            filepath = Path(_raw)
    elif len(widget_selected_file.value) == 1:
        filepath = widget_selected_file.value[0].path

    if filepath:
        mo.output.append(
            mo.md(
                f"""Correctly loaded `{filepath}` from 

            `{filepath.parent}` """
            ).callout(kind="success")
        )
    return (filepath,)


@app.cell
def _():
    # peaks = auto_pick_peaks(
    #     filepath,
    #     min_mz=350.0,
    #     max_mz=1200.0,
    #     bin_width=0.002,
    #     mz_tolerance_absolute=0.005,
    #     mz_tolerance_relative=3e-6,
    #     min_separation_absolute=0.005 * 1.5,
    #     min_relative_intensity=0.005,
    # )
    return


@app.cell
def _(filepath):
    mo.stop(not filepath)

    mo.md("# Select m/z values to export")
    return


@app.cell
def _():
    _tol_steps = np.round(
        np.r_[
            1e-3:1e-2:1e-3, 1e-2:1e-1:1e-2, 1e-1:1.1:1e-1, 1:11:1, 10:110:10
        ],
        decimals=3,
    )

    # slider_single_mz = mo.ui.slider(
    #     start=min_mz_global,
    #     stop=max_mz_global,
    #     step=0.01,
    #     value=((max_mz_global - min_mz_global) // 2) + min_mz_global,
    #     include_input=True,
    #     label="Target m/z",
    # )

    text_single_mz = mo.ui.text(kind="text", label="Single m/z:")

    slider_tol = mo.ui.slider(
        steps=_tol_steps, value=0.0050, show_value=True, label="Tolerance"
    )

    widget_csv_upload = mo.ui.file(
        filetypes=[".csv"],
        kind="button",
        multiple=False,
    )

    widget_use_folder_csv = mo.ui.run_button(
        label="Use CSV from selected file folder"
    )

    _instructions = mo.md(
        """

        ---

        Click the button below and select a single .csv file.

        The file must contain m/z values in a single column, have no header, and use a dot (`.`) as the decimal separator.


        """
    )

    widget_tab_mz = mo.ui.tabs(
        tabs={
            "Multiple m/z": mo.vstack(
                [
                    slider_tol,
                    _instructions,
                    widget_csv_upload,
                    mo.md(
                        "--- \n Or automatically use the only `.csv` in the same folder as the selected `.imzml` file:"
                    ),
                    widget_use_folder_csv,
                ],
                heights="equal",
            ),
            "Single m/z": mo.vstack([slider_tol, text_single_mz]),
        }
    )

    widget_tab_mz
    return (
        slider_tol,
        text_single_mz,
        widget_csv_upload,
        widget_tab_mz,
        widget_use_folder_csv,
    )


@app.cell
def _(filepath):
    from bioio import BioImage

    _pps = BioImage(filepath).physical_pixel_sizes if filepath else None
    pixel_size_missing = _pps is not None and (
        _pps.X is None or _pps.Y is None
    )
    return (pixel_size_missing,)


@app.cell
def _(pixel_size_missing):
    pixel_size_input = mo.ui.text(
        label="Pixel size (µm)", placeholder="e.g. 10"
    )

    mo.vstack(
        [
            mo.md(
                "**No pixel size found in the imzML metadata.** "
                "Enter it below (µm; square pixels assumed)."
            ).callout(kind="warn"),
            pixel_size_input,
        ]
    ) if pixel_size_missing else None
    return (pixel_size_input,)


@app.cell
def _(filepath, widget_csv_upload, widget_use_folder_csv):
    mz_list_from_csv = None
    mz_csv_source = None

    if widget_use_folder_csv.value:
        if filepath is None:
            mo.output.append(
                mo.md("Select an `.imzml` file first.").callout(kind="warn")
            )
        else:
            _csv_candidates = sorted(filepath.parent.glob("*.csv"))

            if len(_csv_candidates) == 1:
                try:
                    mz_list_from_csv = parse_mz_csv(_csv_candidates[0])
                    mz_csv_source = _csv_candidates[0].name
                    mo.output.append(
                        mo.md(
                            f"Using CSV from folder: `{mz_csv_source}`"
                        ).callout(kind="success")
                    )
                except Exception as e:
                    mo.output.append(
                        mo.md(
                            f"Error reading CSV `{_csv_candidates[0].name}`: `{e}`"
                        ).callout(kind="danger")
                    )
            elif len(_csv_candidates) == 0:
                mo.output.append(
                    mo.md(
                        f"No `.csv` file found in `{filepath.parent}`."
                    ).callout(kind="warn")
                )
            else:
                _names = ", ".join(_p.name for _p in _csv_candidates)
                mo.output.append(
                    mo.md(
                        "Expected exactly one `.csv` in the selected file folder, "
                        f"but found `{len(_csv_candidates)}`: `{_names}`"
                    ).callout(kind="warn")
                )

    if len(widget_csv_upload.value) == 1:
        try:
            mz_list_from_csv = parse_mz_csv(
                widget_csv_upload.value[0].contents
            )
            mz_csv_source = widget_csv_upload.value[0].name

        except Exception as e:
            mo.output.append(
                mo.md(f"Error reading CSV: `{e}`").callout(kind="danger")
            )
            mz_list_from_csv = None
            mz_csv_source = None
    return mz_csv_source, mz_list_from_csv


@app.cell
def _(
    is_script_mode,
    mz_csv_source,
    mz_list_from_csv,
    slider_tol,
    text_single_mz,
    widget_tab_mz,
):
    m_z_values = []
    tolerance = slider_tol.value

    if is_script_mode:
        _args = mo.cli_args()
        if "tolerance" in _args:
            tolerance = float(_args["tolerance"])
        if "mz-csv" in _args:
            m_z_values = parse_mz_csv(_args["mz-csv"])
        elif "mz" in _args:
            m_z_values = [float(_args["mz"])]
        else:
            # m_z_values = [slider_single_mz.value]
            m_z_values = [float(text_single_mz.value)]
    elif widget_tab_mz.value == "Single m/z":
        m_z_values = [float(text_single_mz.value)]
        mo.output.append(
            mo.md(
                f"**Single m/z mode selected.** Processing: `{m_z_values[0]:.4f}`"
            ).callout(kind="success")
        )

    elif widget_tab_mz.value == "Multiple m/z":
        if mz_list_from_csv:
            m_z_values = mz_list_from_csv
            _src = mz_csv_source or "uploaded CSV"
            mo.output.append(
                mo.md(
                    f"**Multiple m/z mode selected.** Found `{len(m_z_values)}` m/z values in `{_src}`."
                ).callout(kind="success")
            )
        else:
            mo.output.append(
                mo.md(
                    "**Multiple m/z mode selected, but no valid CSV file loaded.**"
                ).callout(kind="warn")
            )
            m_z_values = []

    # if 760.5851 not in m_z_values:
    #     m_z_values.append(760.5851)
    m_z_values = sorted(m_z_values)
    return m_z_values, tolerance


@app.cell
def _(filepath, m_z_values, pixel_size_input, pixel_size_missing, tolerance):
    mo.stop(
        not m_z_values,
        mo.md("**Waiting for valid m/z input...**").callout(kind="warn"),
    )

    _pixel_size = None
    if pixel_size_missing:
        mo.stop(
            not pixel_size_input.value.strip(),
            mo.md("**Enter the pixel size (µm) above.**").callout(kind="warn"),
        )
        _v = float(pixel_size_input.value)
        _pixel_size = (_v, _v)

    with mo.status.spinner(
        title="Loading...", subtitle="This will take 1-2 minutes"
    ):
        ion_image_xr = read_imzml_to_xarray(
            filepath, m_z_values, tolerance, pixel_size=_pixel_size
        )

    show_export_button = True

    mo.md("# Preview")
    return ion_image_xr, show_export_button


@app.cell
def _(ion_image_xr):
    _pixel_size_y, _pixel_size_x = pixel_sizes(ion_image_xr)

    ion_image_xr.hvplot.image(
        x="X",
        y="Y",
        aspect="equal",
        data_aspect=_pixel_size_y / _pixel_size_x,
        hover=True,
        robust=True,
        cmap="viridis",
    ).opts(toolbar=None)
    return


@app.cell
def _(ion_image_xr):
    _pixel_size_y, _pixel_size_x = pixel_sizes(ion_image_xr)

    if _pixel_size_x != _pixel_size_y:
        mo.output.append(mo.md("Pixels are not square").callout(kind="warn"))

    mo.output.append(
        mo.vstack(
            [
                mo.md("Please ***check*** the extracted parameters:"),
                mo.md(
                    f"""
            | Parameter | Value |
            | ---: | :--- |
            | $x$ pixel size: | ${_pixel_size_x} \\ μm$  |
            | $y$ pixel size: | ${_pixel_size_y} \\ μm$  |
            | min m/z : | {ion_image_xr.M_Z.values.min()} |
            | max m/z : | {ion_image_xr.M_Z.values.max()} |
        """
                ).center(),
            ],
            align="center",
            justify="center",
        ).callout(kind="warn")
    )
    return


@app.cell
def _(show_export_button):
    export_button = mo.ui.run_button(label="Export")

    mo.stop(not show_export_button)
    mo.vstack([mo.md("### Press the following button to save"), export_button])
    return (export_button,)


@app.cell
def _(export_button, filepath, ion_image_xr, is_script_mode):
    mo.stop(not export_button.value and not is_script_mode)

    out_dir = filepath.parent / (
        "original_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    out_dir.mkdir(exist_ok=True, parents=True)

    with mo.status.spinner(
        title="Saving...", subtitle="This should take 1-2 minutes"
    ) as _spinner:
        save_dataarray(
            ion_image_xr,
            out_dir / f"{filepath.name}_original.ome.zarr",
        )

    mo.md(f"""Saved original `{filepath.name}_original.ome.zarr` to:

    `{out_dir}`

    """).callout(kind="success")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
