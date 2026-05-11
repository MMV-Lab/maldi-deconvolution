import marimo

__generated_with = "0.23.1"
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


@app.cell
def _():
    from scipy.signal import convolve2d


    def perform_single_slice_convolution(img_2d, kernel):
        """
        Operates on a single 2D slice at a time.
        Input shape:  (y, x)
        Output shape: (y, x)
        """
        img_2d = img_2d.astype(np.float32)

        # 1. Normalize
        # (Note: Convolution is linear so this isn't strictly necessary like it is
        # for Richardson-Lucy, but we keep it here to match your original logic
        # and prevent potential float overflow).
        max_val = np.max(img_2d)
        if max_val > 0:
            img_2d /= max_val

            # 2. Forward Convolution
            # mode='same' keeps the output dimensions identical to the input
            # boundary='symm' (symmetric) is typical to prevent edge artifacts
            img_2d = convolve2d(img_2d, kernel, mode="same", boundary="symm")

            # 3. Rescale back to original intensity range
            img_2d *= max_val

        return img_2d.astype(np.float32)


    def convolve_dataset(input_xarray, kernel):
        """
        Applies 2D convolution over an xarray dataset slice by slice.
        """
        # Apply Ufunc
        convolved_da = xr.apply_ufunc(
            perform_single_slice_convolution,
            input_xarray,
            kwargs={"kernel": kernel},
            input_core_dims=[["y", "x"]],
            output_core_dims=[["y", "x"]],
            vectorize=True,  # True means xarray feeds strictly 2D slices to the function
            dask="parallelized",  # Still distributes across Dask chunks safely
            output_dtypes=[np.float32],
        )

        # Add metadata
        convolved_da.name = "convolved_ion_intensity"
        convolved_da.attrs["long_name"] = "Convolved Ion Intensity"
        convolved_da.attrs["units"] = input_xarray.attrs.get("units", "a.u.")

        return convolved_da

    return (convolve_dataset,)


@app.cell
def _(deconvolute_dataset, load_decay_profile, original_xr, parameters_form):
    # Extract parameters from the form
    laser_diameter = float(parameters_form.value["laser_diameter"])
    step_size = float(parameters_form.value["step_size"])
    kernel_size = int(parameters_form.value["kernel_size"])
    decay_df_compare, _ = load_decay_profile(parameters_form.value["decay_csv"][0])
    decay_errors_compare = (
        decay_df_compare["error"].to_list()
        if "error" in decay_df_compare.columns
        else None
    )
    scan_direction = parameters_form.value["scan_dir"]
    decay_model = parameters_form.value["decay_model"]

    # 1. GENERATE TWO PSFs: Geometric and Beam Profile
    psf_geometric = MaldiPSFGenerator(
        laser_diameter_um=laser_diameter,
        step_size_um=step_size,
        kernel_size=kernel_size,
        scan_direction=scan_direction,
        decay_factors=decay_df_compare["intensity"].to_list(),
        decay_errors=decay_errors_compare,
        decay_model=decay_model,
        method="geometric",
        sim_res_um=0.01,
    )
    psf_geometric_ds = psf_geometric.to_xarray(fov_only=False)

    psf_beam = MaldiPSFGenerator(
        laser_diameter_um=laser_diameter,
        step_size_um=step_size,
        kernel_size=kernel_size,
        scan_direction=scan_direction,
        decay_factors=decay_df_compare["intensity"].to_list(),
        decay_errors=decay_errors_compare,
        decay_model=decay_model,
        method="beam_profile",
        sim_res_um=0.01,
    )
    psf_beam_ds = psf_beam.to_xarray(fov_only=False)


    # Deconvolve with geometric method
    deconv_geometric = deconvolute_dataset(
        original_xr,
        kernel=psf_geometric_ds.final_signal_kernel.values,
        num_iterations=50,
    )

    # Deconvolve with beam profile method
    deconv_beam = deconvolute_dataset(
        original_xr,
        kernel=psf_beam_ds.final_signal_kernel.values,
        num_iterations=50,
    )
    return deconv_beam, deconv_geometric, psf_beam_ds, psf_geometric_ds


@app.cell
def _(
    convolve_dataset,
    deconv_beam,
    deconv_geometric,
    original_xr,
    psf_beam_ds,
    psf_geometric_ds,
):
    # select only top 10
    original_data = (
        original_xr.sortby(original_xr.median(dim=["x", "y"]), ascending=False)
        .head(m_z=50)
        .copy()
    )

    deconv_geometric_final = deconv_geometric.sel(
        iteration=50, m_z=original_data.m_z
    )
    deconv_beam_final = deconv_beam.sel(iteration=50, m_z=original_data.m_z)

    reconvolved_geometric = convolve_dataset(
        deconv_geometric_final, psf_geometric_ds.final_signal_kernel.values
    )
    reconvolved_beam = convolve_dataset(
        deconv_beam_final, psf_beam_ds.final_signal_kernel.values
    )
    return original_data, reconvolved_beam, reconvolved_geometric


@app.cell
def _(original_data, reconvolved_beam, reconvolved_geometric):
    def calculate_metrics_optimized(true, pred, label):

        residual_sq = (true - pred) ** 2

        sse = residual_sq.sum(dim=["x", "y"])
        mse = residual_sq.mean(dim=["x", "y"])
        rmse = np.sqrt(mse)

        data_mean = true.mean(dim=["x", "y"])
        data_range = true.max(dim=["x", "y"]) - true.min(dim=["x", "y"])
        data_std = true.std(dim=["x", "y"])

        ds_metrics = xr.Dataset(
            {
                "SSE": sse,
                "RMSE": rmse,
                "NRMSE_Mean": rmse / data_mean,
                "NRMSE_Range": rmse / data_range,
                "NRMSE_Std": rmse / data_std,
            }
        )

        df = pl.from_pandas(ds_metrics.to_dataframe().reset_index())

        return df.with_columns(pl.lit(label).alias("Method"))


    df_geometric = calculate_metrics_optimized(
        original_data, reconvolved_geometric, "Geometric"
    )
    df_beam = calculate_metrics_optimized(original_data, reconvolved_beam, "Beam")

    df_metrics = pl.concat([df_geometric, df_beam])
    return (df_metrics,)


@app.cell
def _(df_metrics):
    (
        df_metrics.group_by("Method").agg(
            pl.selectors.numeric().exclude("m_z").median()
        )
    )
    return


if __name__ == "__main__":
    app.run()
