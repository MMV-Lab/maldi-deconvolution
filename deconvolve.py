import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")

with app.setup:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import tifffile
    from skimage.restoration import richardson_lucy
    from pathlib import Path
    import marimo as mo


@app.cell
def _():
    LASER_DIAMETER = 4.0  # µm
    STEP_SIZE = 1.3  # µm
    KERNEL_SIZE = 3  # px

    DECAY_CSV_PATH = (
        "data/Sven_Chiahsin_Deconvolution_Oversampling_MALDI/Signal loss/A43.5.csv"
    )
    INPUT_DIR = "/mnt/eternus/users/Davide/maldi-deconv-rl/data/Sven_Chiahsin_Deconvolution_Oversampling_MALDI/Oversampled_U2OS-FUCCI-Cell/CC257_U2OS_1.3 µm"
    OUTPUT_DIR = "/mnt/eternus/users/Davide/maldi-deconv-rl/output/tmp"
    return (DECAY_CSV_PATH,)


@app.cell
def _(DECAY_CSV_PATH):
    if os.path.exists(DECAY_CSV_PATH):
        decay_df = pd.read_csv(DECAY_CSV_PATH)
        # Normalize intensity relative to the first pass
        decay_df["normalized_intensity"] = (
            decay_df["Average intensity"] / decay_df["Average intensity"].iloc[0]
        )

        # Create a lookup dictionary
        decay_lookup = dict(
            zip(decay_df["Number of passes"], decay_df["normalized_intensity"])
        )
        # Get the last valid decay value for clamping (no extrapolation)
        max_passes_in_data = decay_df["Number of passes"].max()
        last_decay_value = decay_df.loc[
            decay_df["Number of passes"] == max_passes_in_data,
            "normalized_intensity",
        ].values[0]

        print("Loaded Signal Decay Data:")
        print(decay_df[["Number of passes", "normalized_intensity"]].head(10))
    else:
        print(f"Warning: CSV file not found at {DECAY_CSV_PATH}. Using identity decay.")
        decay_lookup = {1: 1.0}
        max_passes_in_data = 1
        last_decay_value = 1.0
    return decay_lookup, last_decay_value, max_passes_in_data


@app.cell
def _(decay_lookup, last_decay_value, max_passes_in_data):
    def get_decay_factor(n_passes):
        # Retrieve decay factor. If n_passes > data, clamp to last known value.
        if n_passes < 1:
            return 1.0
        if n_passes > max_passes_in_data:
            return last_decay_value
        return decay_lookup.get(int(n_passes), last_decay_value)

    v_get_decay = np.vectorize(get_decay_factor)
    return (v_get_decay,)


@app.cell
def _(v_get_decay):
    print(v_get_decay)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
