import marimo

__generated_with = "0.23.16"
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
    from bioio_ome_zarr.writers.metadata import Channel


@app.cell
def _():
    # 1. Prepare sample image data (5D array: TCZYX)
    _data = np.random.randint(0, 65535, size=(1, 2, 5, 512, 512), dtype=np.uint16)

    # 2. Define optional OMERO channel metadata
    _channels = [
        Channel(label="DAPI", color="#0000FF"),
        Channel(label="GFP", color="#00FF00"),
    ]

    _zip_path = "sample_image.ome.zarr.zip"

    # 3. Use ZipStore within a context manager (auto-closes and flushes zip header on exit)
    with zarr.storage.ZipStore(_zip_path, mode="w") as store:
        # 4. Instantiate OMEZarrWriter with required parameters
        writer = OMEZarrWriter(
            store=store,
            level_shapes=_data.shape,  # Required: single tuple or list of shapes for pyramids
            dtype=_data.dtype,          # Required: numpy data type
            channels=_channels,
            axes_names=["t", "c", "z", "y", "x"],
            axes_units=[None, None, "micrometer", "micrometer", "micrometer"],
            physical_pixel_size=[1.0, 1.0, 0.5, 0.25, 0.25],
            zarr_format=3,             # 2 for NGFF 0.4, 3 for NGFF 0.5
        )

        # 5. Write the full array volume to the store
        writer.write_full_volume(_data)

    print(f"Successfully wrote OME-Zarr ZIP file to: {_zip_path}")
    return


@app.cell
def _():
    _level_shapes = [
        (2, 3, 4, 256, 256),  # L0 full res
        (2, 3, 4, 128, 128),  # L1 downsampled Y/X by 2
    ]

    _data = np.random.randint(0, 255, size=_level_shapes[0], dtype=np.uint8)
    _channels = [Channel(label=f"c{i}", color="FF0000") for i in range(_data.shape[1])]


    _writer = OMEZarrWriter(
        store="test.ome.zarr.zip",
        level_shapes=_level_shapes,
        dtype=_data.dtype,
        zarr_format=3,
        channels=_channels,
        axes_names=["t", "c", "z", "y", "x"],
        axes_types=["time", "channel", "space", "space", "space"],
        axes_units=[None, None, "micrometer", "micrometer", "micrometer"],
    )

    _writer.write_full_volume(_data)
    return


@app.cell
def _():
    filepath = r"/mnt/eternus/users/Davide/maldi-deconv-rl/data/images/SMALDI 1.4/U2OS-FUCCI/015-2/deconvolved_2026-05-26_13-58-19/Exploris240-2_CC015-2_U2OS-FUCCI_DHAP_900nm_277x333_2-5kV.imzml_deconvolved.h5"
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
    data_xr.dims
    return


app._unparsable_cell(
    r"""
    def convert_to_zarr(img: xarray.DataArray):
    """,
    name="_"
)


@app.cell
def _(xarray):
    def convert_to_zarr(img: xarray.DataArray):
        if not ("iteration" in img.dims and "m_z" in img.dims):
            raise ValueError("Improper xarray")

        mz_channels = [f"m/z={mz:.6f}" for mz in img.m_z.values]
        print(mz_channels)




    return


@app.cell
def _():
    # 
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
def _(da, mz_channels_raw):
    # --- [Step 1: Your working BioImage setup] ---

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
    return img_xr, zarr_output_path, zarr_writer


@app.cell
def _(da, img_xr, zarr_output_path, zarr_writer):
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
    return


@app.cell
def _(img_xr):
    img_xr.data
    return


if __name__ == "__main__":
    app.run()
