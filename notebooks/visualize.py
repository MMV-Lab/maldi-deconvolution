import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    import numpy as np
    import xarray as xr

    import tifffile
    from pathlib import Path

    import matplotlib as mpl
    import panel as pn
    import holoviews as hv
    import hvplot
    import hvplot.xarray
    from datashader.transfer_functions import shade, stack
    from matplotlib.colors import LinearSegmentedColormap

    from PIL import Image, ImageChops


@app.cell
def _():
    mo.md(r"""
    # Welcome!

    ## This notebook serves the purpose of exporting useful visualizations and raw image data from an already processed `h5` file.
    """)
    return


@app.cell
def _():
    widget_selected_file = mo.ui.file_browser(
        filetypes=[".h5"], multiple=False, initial_path="data"
    )
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
        if "X" in data_xr.dims:
            data_xr = data_xr.rename({"X": "x"})
        if "Y" in data_xr.dims:
            data_xr = data_xr.rename({"Y": "y"})
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
    colormap_names = [
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        "gray",
    ]
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

    linear_configs = {
        "Linear Red": ["black", "red"],
        "Linear Green": ["black", "#00ff00"],
        "Linear Blue": ["black", "blue"],
        "Linear Cyan": ["black", "cyan"],
        "Linear Magenta": ["black", "magenta"],
        "Linear Yellow": ["black", "yellow"],
    }

    all_names = list(linear_configs.keys()) + colormap_names

    cmap_dict = {}
    for name in all_names:
        if name in linear_configs:
            cmap_dict[name] = LinearSegmentedColormap.from_list(
                name, linear_configs[name]
            )
        else:
            cmap_dict[name] = mpl.colormaps[name]
    return cmap_dict, colormap_names


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
        name="Upper %", start=80, end=100.0, value=99.8
    )
    _cmap_list = (
        list(cmap_dict.keys())
        if isinstance(cmap_dict, dict)
        else ["viridis", "plasma", "Blues", "RdBu_r"]
    )
    cmap_widget = pn.widgets.Select(
        name="Colormap", options=_cmap_list, value="viridis"
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

        if has_iteration and iteration is not None:
            slice_data = data_xr.sel(m_z=m_z, iteration=iteration)
            title = f"m/z = {m_z:.4f} , iteration = {iteration}"
        else:
            slice_data = data_xr.sel(m_z=m_z)
            title = f"m/z = {m_z:.4f}"

        v_min, v_max = np.nanpercentile(slice_data.values, [low_p, high_p])

        if v_max <= v_min:
            v_max = v_min + 1e-6

        _aspect = float(
            data_xr.attrs.get("pixel_size_y")
            / data_xr.attrs.get("pixel_size_x")
        )

        return slice_data.hvplot.image(
            x="x",
            y="y",
            # aspect=1,
            data_aspect=_aspect,
            width=700,
            clim=(v_min, v_max),
            cmap=cmap_name,
            colorbar=True,
            responsive=False,
            title=title,
        ).opts(invert_yaxis=True)

    def export_datacube_tiff(event):
        status_text.value = "⏳ Exporting TIFFs... please wait."
        export_tiff_button.disabled = True

        try:
            # 1. Metadata Extraction
            pixel_size_x = data_xr.attrs.get("pixel_size_x", 1.0)
            pixel_size_y = data_xr.attrs.get("pixel_size_y", 1.0)
            original_file_name = data_xr.attrs.get(
                "original_file_name", "data"
            )
            tolerance = data_xr.attrs.get("tolerance", 0.0)

            tiff_dir = out_dir / "exported_datacube"
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
        status_text.value = "⏳ Saving high-res plots (SVG/PNG), RGB TIFF, and raw slice TIFF..."

        try:
            # ============= PLOTS

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

            hv.save(current_plot, svg_path, backend="matplotlib", dpi=600)

            hv.save(current_plot, png_path, backend="matplotlib", dpi=600)

            # ============= TIFF

            data_2d = current_plot.data[list(current_plot.data.data_vars)[0]]
            pixel_size_x = data_2d.attrs.get("pixel_size_x")
            pixel_size_y = data_2d.attrs.get("pixel_size_y")

            v_min, v_max = np.nanpercentile(
                data_2d.values, [low_perc_slider.value, high_perc_slider.value]
            )

            shaded = shade(
                data_2d,
                cmap=cmap_dict[cmap_widget.value],
                span=(v_min, v_max),
                how="linear",
            )

            rgb_array = np.array(shaded.to_pil().convert("RGB"))
            rgb_array = np.flipud(rgb_array)

            tiff_path = out_dir / f"{base_name}_render.tiff"
            tifffile.imwrite(
                tiff_path,
                rgb_array,
                imagej=True,
                compression="zlib",
                photometric="rgb",
                resolution=(1 / pixel_size_x, 1 / pixel_size_y),
                metadata={
                    "axes": "CYX",
                    "unit": "um",
                },
            )

            # Save the raw numeric slice (float32), similar to the datacube export.
            raw_slice = (
                data_xr.sel(m_z=m_z_widget.value, iteration=iter_val)
                if iter_val is not None
                else data_xr.sel(m_z=m_z_widget.value)
            )
            raw_tiff_path = out_dir / f"{base_name}_raw.tiff"
            tifffile.imwrite(
                raw_tiff_path,
                raw_slice.values.astype(np.float32),
                imagej=True,
                compression="zlib",
                photometric="minisblack",
                resolution=(1 / pixel_size_x, 1 / pixel_size_y),
                metadata={
                    "axes": "YX",
                    "unit": "um",
                },
            )

            status_text.value = "✅ Successfully saved SVG, PNG, RGB-TIFF, and raw slice TIFF to output folder."

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
        mz_options,
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
    layout
    return


@app.cell
def _():
    return


@app.cell
def _(data_xr, mz_options):
    _tmp = data_xr.isel(iteration=-1).median(dim=["x", "y"])
    _mz_intensity_map = dict(zip(_tmp.m_z.values, _tmp.values))
    _sorted_keys = sorted(
        mz_options.keys(),
        key=lambda k: _mz_intensity_map.get(mz_options[k], 0),
        reverse=True,
    )

    # 3. Access your top 3
    _top_3_keys = _sorted_keys[:3]

    m_z_multichannel = mo.ui.multiselect(
        mz_options, value=_top_3_keys, label="m/z:"
    )

    iteration_multichannel = mo.ui.slider(
        steps=data_xr.iteration.values,
        show_value=True,
        label="Iteration",
        value=max(data_xr.iteration.values),
    )

    mo.vstack([m_z_multichannel, iteration_multichannel])
    return iteration_multichannel, m_z_multichannel


@app.cell
def _(cmap_dict, m_z_multichannel):
    colormap_multichannel = mo.ui.array(
        [
            mo.ui.dropdown(
                options=cmap_dict,
                searchable=True,
                label=f"Ch{i} (m/z {el:.4f})",
            )
            for i, el in enumerate(m_z_multichannel.value)
        ]
    )

    low_perc_multichannel = mo.ui.array(
        [
            mo.ui.slider(
                steps=np.arange(start=0, stop=99),
                value=0,
                show_value=True,
                label="min %",
            )
            for i, el in enumerate(m_z_multichannel.value)
        ]
    )

    high_perc_multichannel = mo.ui.array(
        [
            mo.ui.slider(
                steps=np.round(np.arange(95, 100.1, 0.1), 2),
                value=99.8,
                show_value=True,
                label="max %",
            )
            for i, el in enumerate(m_z_multichannel.value)
        ]
    )

    run_multichannel = mo.ui.run_button(label="Composite")

    mo.vstack(
        [
            mo.hstack(
                [
                    colormap_multichannel.vstack(),
                    low_perc_multichannel.vstack(),
                    high_perc_multichannel.vstack(),
                ]
            ),
            run_multichannel,
        ]
    )
    return colormap_multichannel, high_perc_multichannel, low_perc_multichannel


@app.cell
def _(
    colormap_multichannel,
    data_xr,
    high_perc_multichannel,
    iteration_multichannel,
    low_perc_multichannel,
    m_z_multichannel,
):
    multichannel_imgs_list = []
    single_channels_imgs_list = []

    for i in range(len(m_z_multichannel.value)):
        _data = data_xr.sel(
            m_z=m_z_multichannel.value[i],
            iteration=iteration_multichannel.value,
        )
        _span = np.nanpercentile(
            _data.values,
            [low_perc_multichannel.value[i], high_perc_multichannel.value[i]],
        )
        _span.tolist()
        _ch = shade(
            _data.isel(y=slice(None, None, -1)),  # INVERT THE Y DIRECTION
            cmap=colormap_multichannel.value[i],
            how="linear",
            span=_span,
        )

        multichannel_imgs_list.append(_ch)
        single_channels_imgs_list.append(_ch)

    multichannel_img = stack(*multichannel_imgs_list, how="add")

    mo.output.append(mo.hstack(single_channels_imgs_list))
    mo.output.append(multichannel_img)
    return (multichannel_img,)


@app.cell
def _(multichannel_img):
    _t = multichannel_img.to_pil()
    mo.inspect(_t)
    return


@app.cell
def _():
    # For white backgorund
    # Now standard stacking (over) will work because
    # the backgrounds are transparent rather than white
    # stack(multichannel_img[0], multichannel_img[1])
    return


@app.cell
def _():
    # _pil = [el.to_pil() for el in multichannel_img]
    # ImageChops.multiply(_pil[0], _pil[1]), Image.alpha_composite(_pil[0], _pil[1])
    return


@app.cell
def _():
    return


@app.cell
def _(
    cmap_dict,
    cmap_widget,
    get_image_plot,
    high_perc_slider,
    iteration_widget,
    low_perc_slider,
    m_z_widget,
):
    _current_plot = get_image_plot(
        m_z_widget.value,
        iteration_widget.value,
        low_perc_slider.value,
        high_perc_slider.value,
        cmap_widget.value,
    )

    # 1. Get the data
    # Note: Ensure you are selecting the specific variable if it's a Dataset
    data_2d = _current_plot.data.to_array().squeeze(drop=True).values

    # 2. Normalize and create 8-bit indices
    v_min, v_max = np.nanpercentile(
        data_2d, [low_perc_slider.value, high_perc_slider.value]
    )
    indexed_data = np.interp(data_2d, (v_min, v_max), (0, 255))
    indexed_data = np.nan_to_num(indexed_data, nan=0)
    indexed_data = np.clip(indexed_data, 0, 255).astype(np.uint8)

    # 3. Extract the Colormap
    cmap_obj = cmap_dict[cmap_widget.value]  # The ListedColormap object

    # Instead of cmap_obj.colors, sample it at 256 points.
    # This ensures a smooth ramp even if the ListedColormap only had 10 colors.
    x_steps = np.linspace(0, 1, 256)
    color_floats = cmap_obj(x_steps)[
        :, :3
    ]  # Returns (256, 3) in range 0.0 to 1.0

    # 4. Convert to TIFF's 16-bit Palette Format
    # TIFF requires a (3, 256) array of uint16 (0 to 65535)
    # We multiply by 65535 to scale 1.0 to the max 16-bit value
    palette_for_tiff = (color_floats.T * 65535).astype(np.uint16)
    return


@app.cell
def _():
    swatch_button = mo.ui.run_button(label="Generate Colormap Swatches")
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

            fname = f"{original_file_name}_mz_{mz:.4f}±{tolerance}_original.tiff"
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
    # _tmp_plot = get_image_plot(
    #     m_z_widget.value,
    #     iteration_widget.value,
    #     low_perc_slider.value,
    #     high_perc_slider.value,
    #     cmap_widget.value,
    # )

    # hvplot.render(_tmp_plot, backend="matplotlib", fig="svg", dpi=3000)
    return


@app.cell
def _():
    # if iteration_widget:
    #     slice_sel = data_xr.sel(
    #         m_z=m_z_widget.value, iteration=iteration_widget.value
    #     )
    # else:
    #     slice_sel = data_xr.sel(m_z=m_z_widget.value)

    # _plot = slice_sel.hvplot.image(aspect="square", colorbar=True)
    #
    # hvplot.save(_plot, "test.svg", fig="svg", dpi=3000)
    return


if __name__ == "__main__":
    app.run()
