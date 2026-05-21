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
    import zarr

    from bioio import BioImage, PhysicalPixelSizes
    from bioio.array_like_reader import ArrayLikeReader
    from bioio_ome_zarr.writers import OMEZarrWriter, get_default_config_for_viz


@app.cell
def _():
    filepath = r"/mnt/eternus/users/Davide/maldi-deconv-rl/data/images/SMALDI 5/Cerebellum/DHAP_Positive/282-1/original_2026-05-11_09-34-28/QEHF1_CC282-1_MouseCerebellum_DHAP_A43-5_1-3um_400x238_3-5kV.imzml_original.h5"
    return (filepath,)


@app.cell
def _(filepath):
    data_xr = xr.open_dataarray(filepath)
    return (data_xr,)


@app.cell
def _(data_xr):
    print(data_xr)
    return


@app.cell
def _(data_xr):
    data_xr.to_zarr("data/test.zarr")
    return


@app.cell
def _(data_xr):
    da = data_xr
    return (da,)


@app.cell
def _():
    # mz_channels = [f"{mz:.4f} m/z" for mz in da.m_z.values]
    # pixel_sizes = PhysicalPixelSizes(
    #     X=da.attrs.get("pixel_size_x"), 
    #     Y=da.attrs.get("pixel_size_y"), 
    #     Z=None
    # )
    # da_renamed = da.rename({"m_z": "C", "y": "Y", "x": "X"})

    # # Explicitly instruct BioImage which reader class to use
    # img = BioImage(
    #     da_renamed,
    #     reader=ArrayLikeReader,
    #     dim_order="CYX",
    #     channel_names=mz_channels,
    #     physical_pixel_sizes=pixel_sizes
    # )

    # # Inject custom metadata parameters
    # img.reader._all_scenes[0].attrs["maldi_tolerance"] = da.attrs.get("tolerance")
    # img.reader._all_scenes[0].attrs["original_file_name"] = da.attrs.get("original_file_name")
    # img.reader._all_scenes[0].attrs["units"] = da.attrs.get("units")
    return


@app.cell
def _(da):
    from bioio_ome_zarr.writers.metadata import Channel

    # --- [Step 1: Your working BioImage setup] ---
    mz_channels_raw = [f"{mz:.4f} m/z" for mz in da.m_z.values]
    pixel_sizes = PhysicalPixelSizes(
        X=da.attrs.get("pixel_size_x"), 
        Y=da.attrs.get("pixel_size_y"), 
        Z=None
    )
    da_renamed = da.rename({"m_z": "C", "y": "Y", "x": "X"})

    img = BioImage(
        da_renamed,
        dim_order="CYX",
        channel_names=mz_channels_raw,
        physical_pixel_sizes=pixel_sizes
    )

    # --- [Step 2: Setup OME-Zarr Export via processing.py pattern] ---
    zarr_output_path = Path("output_maldi_image.ome.zarr")
    img_xr = img.xarray_data

    dims_upper = [d.upper() for d in img_xr.dims]       # e.g., ['T', 'C', 'Z', 'Y', 'X']
    axes_names = [d.lower() for d in dims_upper]        # e.g., ['t', 'c', 'z', 'y', 'x']

    physical_pixel_size = [
        getattr(img.scale, dim, 1.0) or 1.0
        for dim in list(img_xr.dims)
    ]

    # FIX: Map properties strictly to 'label' and 'color' keywords
    # Setting the color parameter lets viewers like Vizarr render your coordinates correctly.
    mz_channels_structured = [
        Channel(
            label=ch_name,   # Correct parameter for your "302.4000 m/z" string
            color="FFFFFF",  # Default hex color (White) for rendering windows
        ) 
        for ch_name in mz_channels_raw
    ]

    # Generate optimal multiscale resolutions and chunk maps (~16MiB per block target)
    # downsample_z=False keeps the 2D surface project dimensions clean from vertical interpolation
    viz_config = get_default_config_for_viz(img_xr.data, downsample_z=False)

    # --- [Step 3: Execute the Writer] ---
    zarr_writer = OMEZarrWriter(
        store=zarr_output_path,
        zarr_format=3,                                # Triggers modern OME-NGFF v0.5
        level_shapes=viz_config["level_shapes"],
        chunk_shape=viz_config["chunk_shape"],
        dtype=viz_config["dtype"],
        axes_names=axes_names,
        channels=mz_channels_structured,              # Hand over the clean metadata objects
        physical_pixel_size=physical_pixel_size,
    )

    # Safely write out the active dask array volume
    zarr_writer.write_full_volume(img_xr.data)

    # --- [Step 4: Inject Custom MALDI Metadata directly into the Zarr Store] ---
    root = zarr.open_group(zarr_output_path, mode="a")

    custom_metadata = {
        "maldi_tolerance": da.attrs.get("tolerance"),
        "original_file_name": da.attrs.get("original_file_name"),
        "units": da.attrs.get("units"),
        "long_name": da.attrs.get("long_name"),
    }

    root.attrs.update(custom_metadata)
    return (img_xr,)


@app.cell
def _(img_xr):
    img_xr.data

    return


if __name__ == "__main__":
    app.run()
