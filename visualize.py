import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import numpy as np
    import xarray as xr

    import os
    import tifffile
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib as mpl
    import panel as pn
    import holoviews as hv
    import hvplot
    import hvplot.xarray
    from datashader.transfer_functions import shade


@app.cell
def _():
    mo.md(r"""
    # Welcome!

    ## This notebook serves the purpose of exporting useful visualizations and raw image data from an already processed `h5` file.
    """)
    return


@app.cell
def _():
    widget_selected_file = mo.ui.file_browser(filetypes=[".h5"], multiple=False)
    mo.vstack(
        [
            mo.md(
                "## Select an `h5` file. Both deconvolved and original data are accepted"
            ),
            widget_selected_file,
        ]
    )
    return (widget_selected_file,)


@app.cell
def _(widget_selected_file):
    if len(widget_selected_file.value) == 1:
        input_file = widget_selected_file.value[0].path
        out_dir = input_file.parent
        data_xr = xr.open_dataarray(input_file)
        mo.output.append(
            mo.md(
                f"""Correctly loaded `{input_file.name}` from 

            `{input_file.parent}` """
            ).callout(kind="success")
        )
    return data_xr, out_dir


@app.cell
def _():
    return


@app.cell
def _():
    colormap_names = ["viridis", "plasma", "inferno", "magma", "cividis"]
    colormap_names.extend(
        [
            "Greys",
            "Purples",
            "Blues",
            "Greens",
            "Oranges",
            "Reds",
            "YlOrBr",
            "YlOrRd",
            "OrRd",
            "PuRd",
            "RdPu",
            "BuPu",
            "GnBu",
            "PuBu",
            "YlGnBu",
            "PuBuGn",
            "BuGn",
            "YlGn",
        ]
    )
    return (colormap_names,)


@app.cell
def _(colormap_names):
    cmap_dict = {name: mpl.colormaps[name] for name in colormap_names}
    # cmap_dict = {name: name for name in colormap_names}
    # cmap_dict
    return (cmap_dict,)


@app.cell
def _():
    # nearest_val = data_xr.sel(m_z=720.59, method="nearest").m_z
    # index = np.where(data_xr.m_z == nearest_val)[0][0]
    # print(index)
    return


@app.cell
def _(cmap_dict, data_xr, out_dir):
    mz_vals = data_xr.m_z.values
    mz_options = {f"{val:.4f}": val for val in mz_vals}

    m_z_widget = pn.widgets.DiscreteSlider(
        name="m/z", options=mz_options, value=mz_vals[0], width=400
    )

    # Check for iteration dimension
    has_iteration = "iteration" in data_xr.dims
    if has_iteration:
        iteration_vals = list(data_xr.iteration.values)
        iteration_widget = pn.widgets.DiscreteSlider(
            name="Iteration",
            options=iteration_vals,
            value=int(iteration_vals[0]),
        )
    else:
        iteration_widget = None

    low_perc_slider = pn.widgets.FloatSlider(
        name="Lower %", start=0, end=99, value=0
    )
    high_perc_slider = pn.widgets.FloatSlider(
        name="Upper %", start=98.0, end=100.0, value=99.8
    )
    _cmap_list = (
        list(cmap_dict.keys())
        if isinstance(cmap_dict, dict)
        else ["viridis", "plasma", "Blues", "RdBu_r"]
    )
    cmap_widget = pn.widgets.Select(
        name="Colormap", options=_cmap_list, value="inferno"
    )

    save_button = pn.widgets.Button(
        name="💾 Save High-Res plots", button_type="primary"
    )

    export_tiff_button = pn.widgets.Button(
        name="📦 Export Full Datacube (TIFF)",
        button_type="warning",  # Distinguish it from the plot save
        width=250,
    )

    status_text = pn.widgets.StaticText(value="")


    def get_image_plot(m_z, iteration, low_p, high_p, cmap_name):
        # Map back to the actual cmap if using a dict
        actual_cmap = (
            cmap_dict[cmap_name] if isinstance(cmap_dict, dict) else cmap_name
        )

        if has_iteration and iteration is not None:
            slice_data = data_xr.sel(m_z=m_z, iteration=iteration)
        else:
            slice_data = data_xr.sel(m_z=m_z)

        v_min, v_max = np.nanpercentile(slice_data.values, [low_p, high_p])

        if v_max <= v_min:
            v_max = v_min + 1e-6

        return slice_data.hvplot.image(
            x="x",
            y="y",
            aspect=1,
            frame_width=500,
            clim=(v_min, v_max),
            cmap=actual_cmap,
            colorbar=True,
            responsive=True,
        )


    def export_datacube_tiff(event):
        status_text.value = "⏳ Exporting TIFFs... please wait."
        export_tiff_button.disabled = True

        try:
            # 1. Metadata Extraction
            pixel_size_x = data_xr.attrs.get("pixel_size_x", 1.0)
            pixel_size_y = data_xr.attrs.get("pixel_size_y", 1.0)
            original_file_name = data_xr.attrs.get("original_file_name", "data")
            tolerance = data_xr.attrs.get("tolerance", 0.0)

            tiff_dir = out_dir / "exported_tiffs"
            tiff_dir.mkdir(exist_ok=True)

            total_mz = len(mz_vals)

            for i, mz in enumerate(mz_vals):
                status_text.value = (
                    f"⏳ Processing m/z {mz:.4f} ({i + 1}/{total_mz})"
                )

                if has_iteration:
                    # Select stack: (iteration, y, x)
                    img_stack = data_xr.sel(m_z=mz).values
                    meta_axes = "TYX"
                    fname = f"{original_file_name}_mz_{mz:.4f}±{tolerance}_deconvolved.tiff"
                else:
                    # Select slice: (y, x)
                    img_stack = data_xr.sel(m_z=mz).values
                    meta_axes = "YX"
                    fname = f"{original_file_name}_mz_{mz:.4f}±{tolerance}_original.tiff"

                save_path = tiff_dir / fname

                tifffile.imwrite(
                    save_path,
                    img_stack.astype(np.float32),
                    imagej=True,
                    compression="zlib",
                    photometric="minisblack",
                    resolution=(1 / pixel_size_x, 1 / pixel_size_y),
                    metadata={
                        "axes": meta_axes,
                        "unit": "um",
                    },
                )

            status_text.value = (
                f"✅ Success! Saved {len(mz_vals)} files to /exported_tiffs"
            )

        except Exception as e:
            status_text.value = f"❌ Error: {str(e)}"

        finally:
            export_tiff_button.disabled = False


    def save_plot(event):
        save_button.disabled = True
        status_text.value = "⏳ Saving high-res plots (SVG/PNG)..."

        try:
            iter_val = iteration_widget.value if iteration_widget else None

            current_plot = get_image_plot(
                m_z_widget.value,
                iter_val,
                low_perc_slider.value,
                high_perc_slider.value,
                cmap_widget.value,
            )

            base_name = f"plot_mz_{m_z_widget.value:.4f}"
            if iter_val is not None:
                base_name += f"_iter_{iter_val}"

            base_name += (
                f"_low_{low_perc_slider.value}_high_{high_perc_slider.value}"
                f"_cmap_{cmap_widget.value}"
            )

            svg_path = out_dir / f"{base_name}.svg"
            png_path = out_dir / f"{base_name}.png"

            hv.save(current_plot, svg_path, backend="matplotlib", dpi=1200)

            hv.save(current_plot, png_path, backend="matplotlib", dpi=1200)

            status_text.value = (
                f"✅ Successfully saved SVG and PNG to output folder."
            )

        except Exception as e:
            status_text.value = f"❌ Error saving plot: {str(e)}"

        finally:
            save_button.disabled = False
    return (
        cmap_widget,
        export_datacube_tiff,
        export_tiff_button,
        get_image_plot,
        high_perc_slider,
        iteration_widget,
        low_perc_slider,
        m_z_widget,
        save_button,
        save_plot,
        status_text,
    )


@app.cell
def _(
    cmap_widget,
    export_datacube_tiff,
    export_tiff_button,
    get_image_plot,
    high_perc_slider,
    iteration_widget,
    low_perc_slider,
    m_z_widget,
    save_button,
    save_plot,
    status_text,
):
    pn.extension(
        throttled=True,
        defer_load=True,
        global_loading_spinner=True,
        loading_indicator=True,
    )
    hv.extension("matplotlib")

    status_text.value = ""

    save_button.on_click(save_plot)
    export_tiff_button.on_click(export_datacube_tiff)


    interactive_fn = pn.bind(
        get_image_plot,
        m_z=m_z_widget,
        iteration=iteration_widget if iteration_widget else None,
        low_p=low_perc_slider,
        high_p=high_perc_slider,
        cmap_name=cmap_widget,
    )

    # Adapt layout based on widget presence
    control_widgets = [m_z_widget]
    if iteration_widget:
        control_widgets.append(iteration_widget)

    layout = pn.Column(
        pn.Row(
            pn.Column(*control_widgets),
            pn.Column(low_perc_slider, high_perc_slider, cmap_widget),
        ),
        pn.Row(
            save_button,
            export_tiff_button,
            status_text,
        ),
        interactive_fn,
    )
    return (layout,)


@app.cell
def _(layout):
    mo.lazy(layout, show_loading_indicator=True)
    return


@app.cell
def _():
    swatch_button = mo.ui.run_button(label="Generate Colormap Swatches")
    swatch_button
    return (swatch_button,)


@app.cell
def _(
    colormap_names,
    get_image_plot,
    high_perc_slider,
    iteration_widget,
    low_perc_slider,
    m_z_widget,
    swatch_button,
):
    if swatch_button.value:
        swatch_grid = pn.FlexBox(justify_content="center")

        current_mz = m_z_widget.value
        current_iter = iteration_widget.value if iteration_widget else None
        current_low = low_perc_slider.value
        current_high = high_perc_slider.value

        swatches = []

        for cmap in colormap_names:
            p = get_image_plot(
                m_z=current_mz,
                iteration=current_iter,
                low_p=current_low,
                high_p=current_high,
                cmap_name=cmap,
            )

            # Create a small pane for each swatch
            # We use a fixed width so they tile nicely in the FlexBox
            pane = pn.Column(
                f"**{cmap}**",
                pn.pane.HoloViews(p, backend="matplotlib", width=250),
                margin=0,
            )
            swatches.append(pane)

        # 4. Update the grid and turn off loading
        swatch_grid.objects = swatches

        mo.output.append(swatch_grid)
    return


@app.cell
def _():
    # def export_datacube_tiff_old(event):
    #     export_status.value = "⏳ Exporting TIFFs... please wait."
    #     export_tiff_button.disabled = True

    #     try:
    #         # 1. Metadata Extraction
    #         pixel_size_x = data_xr.attrs.get("pixel_size_x", 1.0)
    #         pixel_size_y = data_xr.attrs.get("pixel_size_y", 1.0)
    #         original_file_name = data_xr.attrs.get("original_file_name", "data")
    #         tolerance = data_xr.attrs.get("tolerance", 0.0)

    #         # 1cm = 10,000 um
    #         res_x = 1e4 / pixel_size_x
    #         res_y = 1e4 / pixel_size_y

    #         fname = f"{original_file_name}_tol{tolerance}_deconvolved.tiff"

    #         img_stack = data_xr.transpose("iteration", "m_z", "y", "x").values

    #         tifffile.imwrite(
    #             out_dir / fname,
    #             img_stack.astype(np.float32),
    #             imagej=True,
    #             # bigtiff=True,
    #             compression="zlib",
    #             photometric="minisblack",
    #             resolution=(res_x, res_y),
    #             metadata={
    #                 "axes": "TCYX",
    #                 "unit": "um",
    #             },
    #         )

    #         export_status.value = f"✅ Success!"

    #     except Exception as e:
    #         export_status.value = f"❌ Error: {str(e)}"

    #     finally:
    #         export_tiff_button.disabled = False
    return


@app.cell
def _():
    # # 1. m_z slider with full precision display

    # pn.extension()

    # mz_vals = data_xr.m_z.values
    # mz_options = {f"{val:.4f}": val for val in mz_vals}
    # m_z_widget = pn.widgets.DiscreteSlider(
    #     name="m/z Value", options=mz_options, value=mz_vals[177], width=300
    # )

    # # 2. Iteration slider
    # iteration_widget = pn.widgets.DiscreteSlider(
    #     name="Iteration",
    #     options=list(data_xr.iteration.values),
    #     value=int(data_xr.iteration.values[8]),
    #     width=300,
    # )

    # # 3. HIGH PRECISION CONTRAST SLIDERS
    # # Lower bound: Standard 0-100 range
    # low_perc_slider = pn.widgets.FloatSlider(
    #     name="Lower Percentile",
    #     start=0,
    #     end=99,
    #     value=0,
    #     step=0.1,
    # )

    # # Upper bound: Fine-grained control for the top 2%
    # high_perc_slider = pn.widgets.FloatSlider(
    #     name="Upper Percentile (Fine Tune)",
    #     start=98.0,
    #     end=100.0,
    #     value=99.8,
    #     step=0.1,
    # )

    # # 4. Colormap widget
    # cmap_widget = pn.widgets.ColorMap(
    #     name="Colormap",
    #     options=cmap_dict,
    #     value=cmap_dict["Blues"],
    # )


    # # 5. Reactive Plotting Function
    # @pn.depends(
    #     m_z=m_z_widget,
    #     iteration=iteration_widget,
    #     low_p=low_perc_slider,
    #     high_p=high_perc_slider,
    #     cmap=cmap_widget,
    # )
    # def get_image_plot(m_z, iteration, low_p, high_p, cmap):
    #     # Select the 2D slice (384x384)
    #     slice_data = data_xr.sel(m_z=m_z, iteration=iteration)

    #     v_min, v_max = np.nanpercentile(slice_data.values, [low_p, high_p])

    #     # Ensure v_max > v_min to avoid hvplot errors
    #     if v_max <= v_min:
    #         v_max = v_min + 1e-6

    #     return slice_data.hvplot.image(
    #         x="x",
    #         y="y",
    #         aspect=1,
    #         data_aspect=1,
    #         clim=(v_min, v_max),
    #         cmap=cmap,
    #         colorbar=True,
    #         hover=False,
    #         # responsive=True,
    #     )
    #     # ).opts(active_tools=["wheel_zoom"], scalebar=True)


    # save_button = pn.widgets.Button(
    #     name="💾 Save High-Res plot", button_type="primary"
    # )


    # def save_plot(event):
    #     # 1. Get the current plot by calling your existing function with current widget values
    #     current_plot = get_image_plot(
    #         m_z=m_z_widget.value,
    #         iteration=iteration_widget.value,
    #         low_p=low_perc_slider.value,
    #         high_p=high_perc_slider.value,
    #         cmap=cmap_widget.value,
    #     )

    #     # 2. Save using the Matplotlib backend for high DPI control
    #     # This doesn't change your interactive plot, just the export
    #     filename = (
    #         f"plot_mz_{m_z_widget.value:.2f}_iter_{iteration_widget.value}.svg"
    #     )

    #     # We use the matplotlib renderer specifically for high-quality static export
    #     hv.save(current_plot, filename, backend="matplotlib", dpi=1200)
    #     print(f"Saved to {filename}")


    # save_button.on_click(save_plot)


    # # 6. Layout
    # layout = pn.Column(
    #     pn.Row(
    #         pn.Column(m_z_widget, iteration_widget),
    #         pn.Column(
    #             low_perc_slider,
    #             high_perc_slider,
    #             cmap_widget,
    #         ),
    #     ),
    #     get_image_plot,
    #     save_button,
    # )

    # mo.plain(layout.servable())
    return


@app.cell
def _():
    return


@app.function
def export_xarray_to_tiff(data_array: xr.DataArray, output_dir: Path):
    """
    Exports an xarray DataArray to TIFF, handling both single ion images
    and multi-iteration stacks.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract attributes for resolution calculation
    pixel_size_x = data_array.attrs.get("pixel_size_x", 1.0)
    pixel_size_y = data_array.attrs.get("pixel_size_y", 1.0)
    original_file_name = data_array.attrs.get("original_file_name", "")
    tolerance = data_array.attrs.get("tolerance", 0.0)

    # 1cm = 10,000 um
    res_x = 1e4 / pixel_size_x
    res_y = 1e4 / pixel_size_y

    # Check if 'iteration' dimension exists
    if "iteration" in data_array.dims:
        print("Exporting as multi-page TIFF (iterations stack)...")
        # Loop through m_z to save separate files for each m/z containing all iterations
        for mz in data_array.m_z.values:
            # Select specific mz: shape becomes (iteration, y, x)
            img_stack = data_array.sel(m_z=mz).values

            fname = f"{original_file_name}_mz_{mz:.4f}±{tolerance}_deconvolved.tiff"
            save_path = output_dir / fname

            tifffile.imwrite(
                save_path,
                img_stack.astype(np.float32),
                imagej=True,
                compression="zlib",
                photometric="minisblack",
                resolution=(res_x, res_y),
                metadata={
                    "axes": "TYX",  # Time/Iteration, Y, X
                    "unit": "um",
                },
            )
            print(f"Saved: {fname}")

    else:
        print("Exporting as single-page TIFFs...")
        for mz in data_array.m_z.values:
            # Select specific mz: shape becomes (y, x)
            img_slice = data_array.sel(m_z=mz).values

            fname = (
                f"{original_file_name}_mz_{mz:.4f}±{tolerance}_original.tiff"
            )
            save_path = output_dir / fname

            tifffile.imwrite(
                save_path,
                img_slice.astype(np.float32),
                imagej=True,
                compression="zlib",
                photometric="minisblack",
                resolution=(res_x, res_y),
                metadata={
                    "axes": "YX",
                    "unit": "um",
                },
            )


@app.cell
def _():
    # m_z_widget.value, iteration_widget.value
    return


@app.cell
def _():
    return


@app.cell
def _(data_xr, iteration_widget, m_z_widget):
    if iteration_widget:
        slice_sel = data_xr.sel(
            m_z=m_z_widget.value, iteration=iteration_widget.value
        )
    else:
        slice_sel = data_xr.sel(m_z=m_z_widget.value)

    _plot = slice_sel.hvplot.image(aspect="square", colorbar=True)
    hvplot.render(_plot, backend="matplotlib", fig="svg", dpi=3000)
    hvplot.save(_plot, "test.svg", fig="svg", dpi=3000)
    return


@app.cell
def _():
    #

    # _plot = slice_sel.hvplot.image(aspect="square", colorbar=True)

    # mr = hv.renderer("matplotlib")
    # mr.fig = "svg"
    # mr.dpi = 300
    # mr.dpi = 300
    # mr.show(_plot)
    # # mr.save(slice_sel.hvplot.image(aspect="square"), "ex")


    # # try_this = hv.renderer('matplotlib').get_plot(slice_sel.hvplot.image(aspect="square"))
    return


@app.cell
def _():
    # selection_dict = {
    #     dim.name: val
    #     for dim, val in zip(preview_plot.kdims, preview_plot.current_key)
    # }

    # print(selection_dict)
    return


@app.cell
def _():
    # slice_sel = data_xr.sel(**selection_dict)
    # slice_sel
    return


@app.cell
def _():
    # hv.extension("matplotlib")
    # hv.output(backend="matplotlib", fig="svg")
    # out = slice_sel.hvplot.image(aspect="square")
    # # hvplot.save(out, "test.svg")
    # out
    return


if __name__ == "__main__":
    app.run()
