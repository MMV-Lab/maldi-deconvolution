import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    from pathlib import Path
    import math
    from pyimzml.ImzMLParser import ImzMLParser
    import hvplot
    import xarray as xr
    import hvplot.xarray
    import warnings
    import polars as pl
    import datetime


@app.function
def read_imzml_to_xarray(
    filepath: Path,
    m_z_values: list,
    tolerance: float = 0.005,
) -> xr.DataArray:
    """Read an imzML file and return an xarray DataArray of ion images.

    Parameters
    ----------
    filepath:
        Path to the .imzml file.
    m_z_values:
        List of target m/z values to extract.
    tolerance:
        Half-window (in Da) around each target m/z for intensity summation.

    Returns
    -------
    xr.DataArray
        Shape ``(n_mz, n_y, n_x)``, with x/y coordinates in μm and m/z as the
        first dimension.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imzml = ImzMLParser(filepath)

    # --- pixel size ---
    pixel_size_x = imzml.imzmldict.get("pixel size x")
    pixel_size_y = imzml.imzmldict.get("pixel size y")
    if pixel_size_x is None or pixel_size_y is None:
        # Try to compute from max dimension and pixel count
        max_dim_x = imzml.imzmldict.get("max dimension x")
        max_dim_y = imzml.imzmldict.get("max dimension y")
        max_pix_x = imzml.imzmldict.get("max count of pixels x")
        max_pix_y = imzml.imzmldict.get("max count of pixels y")

        if not all([max_dim_x, max_dim_y, max_pix_x, max_pix_y]):
            raise ValueError(
                f"Cannot determine pixel size. Missing metadata: "
                f"max_dim_x={max_dim_x}, max_dim_y={max_dim_y}, "
                f"max_pix_x={max_pix_x}, max_pix_y={max_pix_y}"
            )
        pixel_size_x = max_dim_x / max_pix_x
        pixel_size_y = max_dim_y / max_pix_y
    pixel_size_x = float(pixel_size_x)
    pixel_size_y = float(pixel_size_y)

    # --- spatial extent (imzML coordinates are 1-based) ---
    max_x = max(c[0] for c in imzml.coordinates)
    max_y = max(c[1] for c in imzml.coordinates)

    # --- extent m_z ---

    min_mz_global = float("inf")
    max_mz_global = float("-inf")

    # Scan all spectra to determine the full ranges
    # (This iterates over all pixels, so it might take a few seconds)
    for i in range(len(imzml.coordinates)):
        _mzs, _ = imzml.getspectrum(i)
        if len(_mzs) > 0:
            # m/z arrays are typically sorted
            local_min = _mzs[0]
            local_max = _mzs[-1]

            if local_min < min_mz_global:
                min_mz_global = local_min
            if local_max > max_mz_global:
                max_mz_global = local_max

    min_mz_global = math.floor(min_mz_global)
    max_mz_global = math.ceil(max_mz_global)
    # print(f"Global Min m/z: {min_mz_global}")
    # print(f"Global Max m/z: {max_mz_global}")

    # --- validate m/z values are within range ---
    out_of_range = [
        mz for mz in m_z_values if mz < min_mz_global or mz > max_mz_global
    ]
    if out_of_range:
        raise ValueError(
            f"The following m/z values are outside the data range [{min_mz_global}, {max_mz_global}]: {out_of_range}"
        )

    # --- build ion-image stack ---
    targets = np.array(m_z_values)
    lower_bounds = targets - tolerance
    upper_bounds = targets + tolerance
    n_targets = len(targets)

    ion_image_stack = np.zeros((n_targets, max_y, max_x), dtype=np.float32)

    for idx, (x, y, _z) in enumerate(imzml.coordinates):
        x_pos, y_pos = x - 1, y - 1
        mzs, ints = imzml.getspectrum(idx)
        idx_l = np.searchsorted(mzs, lower_bounds, side="left")
        idx_r = np.searchsorted(mzs, upper_bounds, side="right")
        for t in range(n_targets):
            l, r = idx_l[t], idx_r[t]
            if r > l:
                ion_image_stack[t, y_pos, x_pos] = np.sum(ints[l:r])

    # --- wrap in xarray ---
    x_coords_um = (np.arange(max_x) + 1) * pixel_size_x
    y_coords_um = (np.arange(max_y) + 1) * pixel_size_y

    da = xr.DataArray(
        ion_image_stack,
        coords={"m_z": m_z_values, "y": y_coords_um, "x": x_coords_um},
        dims=("m_z", "y", "x"),
        attrs={
            "tolerance": tolerance,
            "pixel_size_x": pixel_size_x,
            "pixel_size_y": pixel_size_y,
            "original_file_name": Path(filepath).name,
        },
        name="ion_intensity",
    )
    da.x.attrs["units"] = "μm"
    da.y.attrs["units"] = "μm"
    da.m_z.attrs["long_name"] = "m/z"
    da.attrs["long_name"] = "Ion Intensity"
    da.attrs["units"] = "a.u."
    return da


@app.cell
def _():
    mo.md(r"""
    # Welcome!

    ## This notebook serves the purpose of extracting data from a `imzML` file to a `h5` file which can be processed later (deconvolution or visualization)
    """)
    return


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _():
    widget_selected_file = mo.ui.file_browser(filetypes=[".imzml"], multiple=False)
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
def _(filepath):
    mo.stop(not filepath)

    mo.md("# Select m/z values to export")
    return


@app.cell
def _(max_mz_global, min_mz_global):
    _tol_steps = np.round(
        np.r_[1e-3:1e-2:1e-3, 1e-2:1e-1:1e-2, 1e-1:1.1:1e-1, 1:11:1, 10:110:10],
        decimals=3,
    )

    slider_single_mz = mo.ui.slider(
        start=min_mz_global,
        stop=max_mz_global,
        step=0.01,
        value=((max_mz_global - min_mz_global) // 2) + min_mz_global,
        include_input=True,
        label="Target m/z",
    )
    slider_tol = mo.ui.slider(
        steps=_tol_steps, value=0.0050, show_value=True, label="Tolerance"
    )

    widget_csv_upload = mo.ui.file(
        filetypes=[".csv"],
        kind="button",
        multiple=False,
    )

    _instructions = mo.md(
        """

        Click the button below and select a single .csv file.

        The file must contain m/z values in a single column, have no header, and use a dot (`.`) as the decimal separator."""
    )

    widget_tab_mz = mo.ui.tabs(
        tabs={
            "Multiple m/z": mo.vstack(
                [slider_tol, _instructions, widget_csv_upload], heights="equal"
            ),
            "Single m/z": mo.vstack([slider_tol, slider_single_mz]),
        }
    )

    widget_tab_mz
    return slider_single_mz, slider_tol, widget_csv_upload, widget_tab_mz


@app.cell
def _(widget_csv_upload):
    mz_list_from_csv = None

    if len(widget_csv_upload.value) == 1:
        try:
            _df = pl.read_csv(
                source=widget_csv_upload.value[0].contents,
                has_header=False,
                new_columns=["m_z"],
            )
            mz_list_from_csv = _df["m_z"].cast(pl.Float64).to_list()

        except Exception as e:
            mo.output.append(
                mo.md(f"Error reading CSV: `{e}`").callout(kind="danger")
            )
            mz_list_from_csv = None
    return (mz_list_from_csv,)


@app.cell
def _(
    is_script_mode,
    mz_list_from_csv,
    slider_single_mz,
    slider_tol,
    widget_tab_mz,
):
    m_z_values = []
    tolerance = slider_tol.value

    if is_script_mode:
        _args = mo.cli_args()
        if "tolerance" in _args:
            tolerance = float(_args["tolerance"])
        if "mz-csv" in _args:
            _df = pl.read_csv(
                _args["mz-csv"], has_header=False, new_columns=["m_z"]
            )
            m_z_values = _df["m_z"].cast(pl.Float64).to_list()
        elif "mz" in _args:
            m_z_values = [float(_args["mz"])]
        else:
            m_z_values = [slider_single_mz.value]
    elif widget_tab_mz.value == "Single m/z":
        m_z_values = [slider_single_mz.value]
        mo.output.append(
            mo.md(
                f"**Single m/z mode selected.** Processing: `{m_z_values[0]:.4f}`"
            ).callout(kind="success")
        )

    elif widget_tab_mz.value == "Multiple m/z":
        if mz_list_from_csv:
            m_z_values = mz_list_from_csv
            mo.output.append(
                mo.md(
                    f"**Multiple m/z mode selected.** Found `{len(m_z_values)}` m/z values in CSV."
                ).callout(kind="success")
            )
        else:
            mo.output.append(
                mo.md(
                    "**Multiple m/z mode selected, but no valid CSV file loaded.**"
                ).callout(kind="warn")
            )
            m_z_values = []

    if 760.5851 not in m_z_values:
        m_z_values.append(760.5851)
    m_z_values = sorted(m_z_values)
    return m_z_values, tolerance


@app.cell
def _(filepath, m_z_values, tolerance):
    mo.stop(
        not m_z_values,
        mo.md("**Waiting for valid m/z input...**").callout(kind="warn"),
    )

    with mo.status.spinner(
        title="Loading...", subtitle="This will take 1-2 minutes"
    ):
        ion_image_xr = read_imzml_to_xarray(filepath, m_z_values, tolerance)

    show_export_button = True

    mo.md("# Preview")
    return ion_image_xr, show_export_button


@app.cell
def _(ion_image_xr):
    ion_image_xr.hvplot.image(
        x="x",
        y="y",
        aspect="equal",
        data_aspect=ion_image_xr.attrs["pixel_size_y"]
        / ion_image_xr.attrs["pixel_size_x"],
        hover=True,
        robust=True,
        cmap="viridis",
    ).opts(toolbar=None)
    return


@app.cell
def _(ion_image_xr):
    ion_image_xr.attrs

    _pixel_size_x = ion_image_xr.attrs["pixel_size_x"]
    _pixel_size_y = ion_image_xr.attrs["pixel_size_y"]


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
            | min m/z : | {ion_image_xr.m_z.values.min()} |
            | max m/z : | {ion_image_xr.m_z.values.max()} |   
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
        ion_image_xr.to_netcdf(out_dir / f"{filepath.name}_original.h5")

    mo.md(f"""Saved original `{filepath.name}_original.h5` to:

    `{out_dir}`

    """).callout(kind="success")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
