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
    from bioio_ome_zarr.writers import (
        OMEZarrWriter,
        get_default_config_for_viz,
    )
    from bioio_ome_zarr.writers.metadata import Channel





if __name__ == "__main__":
    app.run()
