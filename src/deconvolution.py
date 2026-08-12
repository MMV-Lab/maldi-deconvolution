"""Decay-profile CSV loading and Richardson-Lucy batch deconvolution shared
by the ``deconvolve`` notebook.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import polars as pl
import xarray as xr
from skimage.restoration import richardson_lucy


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
        elif width == 2:
            # Assume col 0 is Pass, col 1 is Intensity
            df = df.select(
                [
                    pl.col(df.columns[0]).alias("pass_idx"),
                    pl.col(df.columns[1]).alias("intensity"),
                ]
            )

        # CASE C: 3+ Columns (Pass, Intensity, Error)
        elif width == 3:
            # Assume col 0 is Pass, col 1 is Intensity, col 2 is Error/Uncertainty
            df = df.select(
                [
                    pl.col(df.columns[0]).alias("pass_idx"),
                    pl.col(df.columns[1]).alias("intensity"),
                    pl.col(df.columns[2]).alias("error"),
                ]
            )

        else:
            return None, "CSV structure invalid."

        # 6. STRICT TYPE CASTING & CLEANUP
        # Drop nulls only on required columns; error column may have partial data
        df = df.drop_nulls(subset=["pass_idx", "intensity"])

        cast_exprs = [
            pl.col("pass_idx").cast(pl.Int64),
            pl.col("intensity").cast(pl.Float64),
        ]
        if "error" in df.columns:
            cast_exprs.append(pl.col("error").cast(pl.Float64))

        df = df.with_columns(cast_exprs)

        # Ensure pass_idx is sorted (optional, but usually desired for decay profiles)
        df = df.sort("pass_idx")

        # Reorder columns
        col_order = ["pass_idx", "intensity"]
        if "error" in df.columns:
            col_order.append("error")
        df = df.select(col_order)

        # Normalization logic depends on number of columns
        if "error" not in df.columns:
            # 1-2 column input: normalize to max
            max_intensity = df["intensity"].max()

            # Check if max is zero to avoid division by zero
            if max_intensity == 0:
                return None, "Maximum intensity is 0; cannot normalize data."

            # Normalize the intensity column
            df = df.with_columns(
                (pl.col("intensity") / max_intensity).alias("intensity")
            )
        else:
            # 3-column input: assume already normalized
            # Verify that the first intensity value is 1.0
            first_intensity = df["intensity"][0]
            tolerance = 1e-6
            if abs(first_intensity - 1.0) > tolerance:
                return (
                    None,
                    f"Validation Error: For 3-column input, intensity must be normalized with intensity[0]=1.0, got {first_intensity:.6f}.",
                )

        return df, None

    except FileNotFoundError:
        return None, f"File not found: {source}"
    except Exception as e:
        return None, f"Error processing file: {str(e)}"


def perform_batch_deconvolution(
    img_batch: np.ndarray, kernel: np.ndarray, num_iterations: int
) -> np.ndarray:
    """
    Operates on a chunk of data.
    Input shape:  (batch, Y, X)  <- corresponds to a chunk of M_Z
    Output shape: (batch, ITERATION, Y, X)
    """
    # 1. Prepare output array
    # Shape: (batch_size, num_iterations + 1, Y, X)
    batch_size, h, w = img_batch.shape
    out_shape = (batch_size, num_iterations + 1, h, w)
    output = np.zeros(out_shape, dtype=np.float32)

    # 2. Iterate over the batch (the M_Z chunk)
    # We loop manually here, which is often faster/safer than np.vectorize
    for i in range(batch_size):
        img_2d = img_batch[i].astype(np.float32)

        # Normalize
        max_val = np.max(img_2d)
        if max_val > 0:
            img_2d /= max_val

        # Store Iteration 0 (Original)
        estimate = img_2d.copy()
        output[i, 0, :, :] = estimate

        # Deconvolution Loop
        for iter_idx in range(1, num_iterations + 1):
            estimate = richardson_lucy(estimate, kernel, num_iter=1)
            output[i, iter_idx, :, :] = estimate.astype(np.float32)

        # Rescale back to original intensity range
        output[i] *= max_val

    return output


def deconvolute_dataset(
    input_xarray: xr.DataArray, kernel: np.ndarray, num_iterations: int = 10
) -> xr.DataArray:
    """Richardson-Lucy deconvolve each ``(Y, X)`` slice of `input_xarray`
    against `kernel`, adding a new ``ITERATION`` dim (0..`num_iterations`)."""
    iter_coords = np.arange(num_iterations + 1)

    deconvolved_da = xr.apply_ufunc(
        perform_batch_deconvolution,
        input_xarray,
        kwargs={"kernel": kernel, "num_iterations": num_iterations},
        input_core_dims=[["Y", "X"]],
        output_core_dims=[["ITERATION", "Y", "X"]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {"ITERATION": len(iter_coords)}},
    )

    deconvolved_da = deconvolved_da.assign_coords(ITERATION=iter_coords)

    deconvolved_da.name = "deconvolved_ion_intensity"
    deconvolved_da.attrs["long_name"] = "Ion Intensity"
    deconvolved_da.attrs["units"] = "a.u."
    return deconvolved_da
