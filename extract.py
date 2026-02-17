import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    from pyimzml.ImzMLParser import ImzMLParser, getionimage, browse
    import hvplot
    import xarray as xr
    import hvplot.xarray
    import warnings
    import polars as pl
    import tifffile
    import datetime


@app.cell
def _():
    mo.md(r"""
    # Welcome!

    ## This notebook serves the purpose of extracting data from a `imzML` file to a `h5` file which can be processed later (deconvolution or visualization)
    """)
    return


@app.cell
def _():
    widget_selected_file = mo.ui.file_browser(filetypes=[".imzml"], multiple=False)
    mo.vstack([mo.md("# Select `imzml` file"), widget_selected_file])
    return (widget_selected_file,)


@app.cell
def _(widget_selected_file):
    imzml_file = None

    if len(widget_selected_file.value) == 1:
        filepath = widget_selected_file.value[0].path
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imzml_file = ImzMLParser(filepath)
        mo.output.append(
            mo.md(
                f"""Correctly loaded `{filepath}` from 

            `{filepath.parent}` """
            ).callout(kind="success")
        )
    return filepath, imzml_file


@app.cell
def _():
    # imzml_file.imzmldict
    return


@app.cell
def _():
    # imzml_file.metadata.pretty()
    return


@app.cell
def _(imzml_file):
    mo.stop(not imzml_file)

    try:
        pixel_size_x = imzml_file.imzmldict.get("pixel size x", 1.0)
    except Exception:
        pixel_size_x = 1.0

    try:
        pixel_size_y = imzml_file.imzmldict.get("pixel size y", 1.0)
    except Exception:
        pixel_size_y = 1.0

    pixel_size_x = float(pixel_size_x)
    pixel_size_y = float(pixel_size_y)

    print(f"\n\n--- Pixel Size ---")
    print(f"x: {pixel_size_x}")
    print(f"y: {pixel_size_y}")

    min_mz_global = float("inf")
    max_mz_global = float("-inf")

    # Scan all spectra to determine the full ranges
    # (This iterates over all pixels, so it might take a few seconds)
    for i in range(len(imzml_file.coordinates)):
        _mzs, _ = imzml_file.getspectrum(i)
        if len(_mzs) > 0:
            # m/z arrays are typically sorted
            local_min = _mzs[0]
            local_max = _mzs[-1]

            if local_min < min_mz_global:
                min_mz_global = local_min
            if local_max > max_mz_global:
                max_mz_global = local_max

        # Progress update every 2000 spectra
        if i % 2000 == 0:
            print(f"Scanned {i} / {len(imzml_file.coordinates)}...", end="\r")

    print(f"\n\n--- Global m/z Range (Processed) ---")
    print(f"Global Min m/z: {min_mz_global:.6f}")
    print(f"Global Max m/z: {max_mz_global:.6f}")
    print(f"\n\n--- Global m/z Range (ROUNDED) ---")
    min_mz_global = int(round(min_mz_global))
    max_mz_global = int(round(max_mz_global))
    print(f"Global Min m/z: {min_mz_global}")
    print(f"Global Max m/z: {max_mz_global}")


    mo.vstack(
        [
            mo.md(
                f"""
            Please ***check*** the extracted parameters:  
        """
            ),
            mo.md(
                f"""
            | Parameter | Value |
            | ---: | :--- |
            | $x$ pixel size: | ${pixel_size_x} \\ μm$  |
            | $y$ pixel size: | ${pixel_size_y} \\ μm$  |
            | min m/z: | {min_mz_global} |
            | max m/z: | {max_mz_global} |   
        """
            ).center(),
        ],
        align="center",
        justify="center",
    ).callout(kind="warn")
    return max_mz_global, min_mz_global, pixel_size_x, pixel_size_y


@app.cell
def _():
    return


@app.cell
def _(imzml_file):
    mo.stop(not imzml_file)

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
def _(mz_list_from_csv, slider_single_mz, slider_tol, widget_tab_mz):
    m_z_values = []
    tolerance = slider_tol.value

    if widget_tab_mz.value == "Single m/z":
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
    return m_z_values, tolerance


@app.cell
def _(imzml_file, m_z_values, tolerance):
    mo.stop(
        not m_z_values,
        mo.md("**Waiting for valid m/z input...**").callout(kind="warn"),
    )

    # imzML coordinates are 1-based
    max_x = max(c[0] for c in imzml_file.coordinates)
    max_y = max(c[1] for c in imzml_file.coordinates)

    targets = np.array(m_z_values)
    lower_bounds = targets - tolerance
    upper_bounds = targets + tolerance
    n_targets = len(targets)

    ion_image_stack = np.zeros((n_targets, max_x, max_y), dtype=np.float32)

    with mo.status.spinner(
        title="Loading...", subtitle="This will take 1-2 minutes"
    ) as _spinner:
        for idx, (x, y, z) in enumerate(imzml_file.coordinates):
            x_pos, y_pos = x - 1, y - 1  # imzML coordinates are 1-based,

            mzs, ints = imzml_file.getspectrum(idx)

            idx_l = np.searchsorted(mzs, lower_bounds, side="left")
            idx_r = np.searchsorted(mzs, upper_bounds, side="right")

            for t in range(n_targets):
                l, r = idx_l[t], idx_r[t]
                if r > l:
                    ion_image_stack[t, y_pos, x_pos] = np.sum(ints[l:r])
    return ion_image_stack, max_x, max_y


@app.cell
def _(
    filepath,
    ion_image_stack,
    m_z_values,
    max_x,
    max_y,
    pixel_size_x,
    pixel_size_y,
    tolerance,
):
    _x_coords_px = np.arange(max_x) + 1
    _y_coords_px = np.arange(max_y) + 1

    _x_coords_um = _x_coords_px * pixel_size_x
    _y_coords_um = _y_coords_px * pixel_size_y

    ion_image_xr = xr.DataArray(
        ion_image_stack,
        coords={"m_z": m_z_values, "y": _y_coords_um, "x": _x_coords_um},
        # coords={"m_z": m_z_values, "y": _y_coords_px, "x": _x_coords_px},
        dims=("m_z", "y", "x"),
        attrs={
            "tolerance": tolerance,
            "pixel_size_x": pixel_size_x,
            "pixel_size_y": pixel_size_y,
            "original_file_name": filepath.name,
        },
        name="ion_intensity",
    )

    ion_image_xr.x.attrs["units"] = "μm"
    ion_image_xr.y.attrs["units"] = "μm"
    ion_image_xr.m_z.attrs["long_name"] = "m/z"

    ion_image_xr.attrs["long_name"] = "Ion Intensity"
    ion_image_xr.attrs["units"] = "a.u."


    show_export_button = True

    mo.md("# Preview")
    return ion_image_xr, show_export_button


@app.cell
def _(ion_image_xr):
    ion_image_xr.hvplot.image(
        x="x",
        y="y",
        aspect="equal",
        hover=True,
        robust=True,
        cmap="viridis",
        frame_height=500,
    ).opts(toolbar=None)
    return


@app.cell(disabled=True)
def _(ion_image_xr):
    _normalized = ion_image_xr / ion_image_xr.sum(dim=["x", "y"])

    _mask = (
        (_normalized > 0).sum(dim=["x", "y"]) / (_normalized).count(dim=["x", "y"])
    ) > 0.05

    _normalized.where(_mask, drop=True).m_z
    return


@app.cell(disabled=True)
def _(ion_image_xr):
    tmp = ion_image_xr.median(dim=["x", "y"]) / ion_image_xr.mean(dim=["x", "y"])

    tmp = tmp.where(tmp > 0, drop=True)
    tmp.m_z
    return


@app.cell(disabled=True)
def _(ion_image_xr):
    _tmp = ion_image_xr.std(dim=["x", "y"])
    _tmp
    return


@app.cell(disabled=True)
def _(ion_image_xr):
    _tmp = ion_image_xr.std(dim=["x", "y"])

    top_indices = _tmp.argsort().values
    top_indices_ordered = sorted(top_indices)

    _new = ion_image_xr.isel(m_z=top_indices_ordered)

    # _new = ion_image_xr.isel({"m_z" : _tmp.argsort().values})

    _new.hvplot.image(
        x="x",
        y="y",
        group_by="m_z",
        aspect="equal",
        hover=True,
        robust=True,
        cmap="viridis",
        frame_height=500,
    ).opts(invert_yaxis=True)

    _tmp
    return


@app.cell(disabled=True)
def _(ion_image_xr):
    # _mask = _normalized.max(dim=["x", "y"]) > 1e-6

    _normalized = ion_image_xr / ion_image_xr.sum(dim=["x", "y"])
    _mask = (
        (_normalized > 0).sum(dim=["x", "y"]) / (_normalized).count(dim=["x", "y"])
    ) > 0.05

    print(_mask.sum())

    _valid_slices = _normalized.where(_mask, drop=True)

    _valid_slices.hvplot.image(
        x="x",
        y="y",
        aspect="equal",
        hover=True,
        robust=True,
        cmap="viridis",
        frame_height=500,
    ).opts(invert_yaxis=True)
    return


@app.cell
def _(show_export_button):
    export_button = mo.ui.run_button(label="Export")

    mo.stop(not show_export_button)
    mo.vstack([mo.md("### Press the following button to save"), export_button])
    return (export_button,)


@app.cell
def _(export_button, filepath, ion_image_xr):
    mo.stop(not export_button.value)

    out_dir = filepath.parent / (
        "original_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    out_dir.mkdir(exist_ok=True, parents=True)


    # out_dir = filepath.parent / "extracted_tiffs"
    # out_dir.mkdir(exist_ok=True, parents=True)

    # # 1cm = 1e4 um
    # tiff_resolution_x = 1e4 / pixel_size_x
    # tiff_resolution_y = 1e4 / pixel_size_y

    # with mo.status.spinner(title="Exporting...") as _spinner:
    #     for mz in ion_image_xr.m_z.values:
    #         img_slice = ion_image_xr.sel(m_z=mz).values

    #         fname = f"{filepath.name}_mz_{mz:.4f}±{tolerance}.tiff"
    #         save_full_path = out_dir / fname

    #         tifffile.imwrite(
    #             save_full_path,
    #             img_slice,
    #             imagej=True,
    #             compression="zlib",
    #             photometric="minisblack",
    #             resolution=(1 / pixel_size_x, 1 / pixel_size_y),
    #             metadata={
    #                 "axes": "YX",
    #                 "unit": "um",
    #                 # "Ranges": np.nanpercentile(img_slice, [0, 99]).tolist(),
    #             },
    #         )
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
