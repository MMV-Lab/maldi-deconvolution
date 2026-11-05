import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    import marimo as mo

    import polars as pl
    import numpy as np
    import xarray as xr

    import tifffile
    from pathlib import Path

    import matplotlib as mpl
    import panel as pn
    import holoviews as hv
    import hvplot
    import hvplot.xarray
    import hvplot.polars
    from datashader.transfer_functions import shade, stack

    from PIL import Image, ImageChops

    from pyimzml.ImzMLParser import ImzMLParser

    import sys
    from pathlib import Path as _Path

    _root_dir = _Path(__file__).resolve().parent.parent
    if str(_root_dir) not in sys.path:
        sys.path.append(str(_root_dir))

    from src.imzml_io import load_dataarray
    from src.plotting import hv_mpl_svg_render, decay_curve_group_overlay

    hv.extension("matplotlib")
    hv.output(backend="matplotlib", fig="svg")


@app.cell(hide_code=True)
def _():
    mo.outline()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Intensity Ratio with respect to peak (manual)
    """)
    return


@app.cell
def _():
    manual_file = Path(
        "data/decay_curve/SMALDI 5/Cerebellum/DHAP_positive_A43.5_20260615_laser realigned/multiple_passes_filtered.ome.zarr"
    ).resolve()

    # The store's time axis comes back as a 0-based ITERATION index (see
    # save_dataarray); here it holds ablation passes, numbered from 1.
    data_xr = load_dataarray(manual_file).sortby("M_Z")
    data_xr = data_xr.rename({"ITERATION": "PASS_NUM"})
    data_xr = data_xr.assign_coords(PASS_NUM=data_xr.PASS_NUM + 1)

    intensity_ratios_xr = data_xr / data_xr.sel(M_Z=760.5851, method="nearest")

    # mean intensity per peak, across all pixels/passes, computed on raw (pre-ratio) data
    intensity_by_mz = (
        data_xr.mean(dim=[d for d in data_xr.dims if d != "M_Z"])
        .compute()
        .to_dataframe(name="mean_intensity")
        .reset_index()
    )
    intensity_order = (
        pl.from_pandas(intensity_by_mz)
        .rename({"M_Z": "m/z"})
        .sort("mean_intensity", descending=True)
    )
    return intensity_order, intensity_ratios_xr


@app.cell
def _(intensity_ratios_xr):
    print(intensity_ratios_xr)
    return


@app.cell
def _(intensity_order, intensity_ratios_xr):
    df = (
        pl.from_pandas(
            intensity_ratios_xr.compute().to_dataframe().reset_index()
        )
        .rename({"ion_intensity": "intensity_ratio"})
        .filter(pl.col("M_Z") != 760.5851)
        .filter(
            pl.col("intensity_ratio").is_finite()
        )  # drop inf/nan from div-by-zero pixels
    )

    pass_summary = (
        df.group_by(["M_Z", "PASS_NUM"])
        .agg(
            pl.col("intensity_ratio").mean().alias("mean_ratio"),
            (
                pl.col("intensity_ratio").std()
                / pl.col("intensity_ratio").count().sqrt()
            ).alias("sem_ratio"),
        )
        .sort(["M_Z", "PASS_NUM"])
    )

    pass_summary = pass_summary.rename({"M_Z": "m/z", "PASS_NUM": "pass_num"})

    pass_summary = pass_summary.join(
        intensity_order, on="m/z", how="left"
    ).sort(["mean_intensity", "pass_num"], descending=[True, False])
    return (pass_summary,)


@app.cell
def _(pass_summary):
    # How much does each lipid's relative intensity (ratio vs 760.5851) vary
    # across ablation passes? For each m/z we take the mean ratio per pass and
    # measure the spread across passes relative to that lipid's own average.
    #
    #   cv_pct         = std(mean_ratio) / mean(mean_ratio) * 100
    #   max_dev_pct    = max|mean_ratio - avg| / avg * 100   (worst single pass)
    #   range_pct      = (max - min) / avg * 100             (full swing)
    #
    # The "did not vary more than XX%" statement is the maximum of max_dev_pct
    # over all lipids (the worst-case single-pass deviation from the mean).
    variation_by_mz = (
        pass_summary.group_by("m/z")
        .agg(
            pl.col("mean_ratio").mean().alias("avg_ratio"),
            pl.col("mean_ratio").std().alias("std_ratio"),
            pl.col("mean_ratio").min().alias("min_ratio"),
            pl.col("mean_ratio").max().alias("max_ratio"),
            pl.col("mean_intensity").first().alias("mean_intensity"),
        )
        .with_columns(
            (pl.col("std_ratio") / pl.col("avg_ratio") * 100).alias("cv_pct"),
            (
                pl.max_horizontal(
                    (pl.col("max_ratio") - pl.col("avg_ratio")).abs(),
                    (pl.col("avg_ratio") - pl.col("min_ratio")).abs(),
                )
                / pl.col("avg_ratio")
                * 100
            ).alias("max_dev_pct"),
            (
                (pl.col("max_ratio") - pl.col("min_ratio"))
                / pl.col("avg_ratio")
                * 100
            ).alias("range_pct"),
        )
        .sort("mean_intensity", descending=True)
    )

    worst_max_dev = variation_by_mz["max_dev_pct"].max()
    worst_cv = variation_by_mz["cv_pct"].max()
    worst_range = variation_by_mz["range_pct"].max()

    print("Relative intensity variation across passes (worst-case lipid):")
    print(f"  max single-pass deviation from mean : {worst_max_dev:.1f}%")
    print(f"  max coefficient of variation (CV)    : {worst_cv:.1f}%")
    print(f"  max full range (max-min)/mean        : {worst_range:.1f}%")

    variation_by_mz
    return


@app.cell
def _(intensity_ratios_xr):
    hv.output(backend="bokeh")

    pass_nums = [1, 3, 5, 9]
    m_z_vals = intensity_ratios_xr.M_Z.values

    grid = hv.GridSpace(kdims=["m/z", "Pass"])

    for p in pass_nums:
        for mz in m_z_vals:
            img_data = intensity_ratios_xr.sel(PASS_NUM=p, M_Z=mz).values

            img = hv.Image(img_data, kdims=["x", "y"]).opts(
                colorbar=False, cmap="viridis"
            )

            grid[mz, p] = img

    grid.opts(
        shared_xaxis=True,
        shared_yaxis=True,
        plot_size=200,
        axiswise=True,
    )

    grid
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # TIC Image of each ablation pass (manual)
    """)
    return


@app.cell
def _():
    import re
    import warnings

    _folder = Path(
        "/mnt/eternus/users/Davide/maldi-deconv-rl/data/decay_curve/SMALDI 5/Cerebellum/DHAP_positive_A43.5"
    )

    _path_list = sorted(
        _folder.glob("*.imzml"),
        key=lambda p: int(re.search(r"(\d+)pass", p.name).group(1)),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _parsers = [ImzMLParser(p) for p in _path_list]

    _n_passes = len(_parsers)
    _coords0 = _parsers[0].coordinates
    _max_x = max(c[0] for c in _coords0)
    _max_y = max(c[1] for c in _coords0)

    _tic_cube = np.zeros((_n_passes, _max_y, _max_x), dtype=np.float32)

    for _p_idx, _parser in enumerate(_parsers):
        for _spec_idx, (_x, _y, _z) in enumerate(_parser.coordinates):
            _mzs, _ints = _parser.getspectrum(_spec_idx)
            _tic_cube[_p_idx, _y - 1, _x - 1] = _ints.sum()

    tic_passes_xr = xr.DataArray(
        _tic_cube,
        coords={
            "PASS_NUM": np.arange(1, _n_passes + 1),
            "Y": np.arange(1, _max_y + 1),
            "X": np.arange(1, _max_x + 1),
        },
        dims=["PASS_NUM", "Y", "X"],
        name="TIC",
    )

    # tic_passes_xr
    return (tic_passes_xr,)


@app.cell
def _():
    return


@app.cell
def _():
    # same 10 peaks / 760.5851
    return


@app.cell
def _(tic_passes_xr):
    _pass_vals = sorted(tic_passes_xr.coords["PASS_NUM"].values.tolist())

    _tic_grid = hv.NdLayout(
        {
            int(p): hv.Image(
                tic_passes_xr.sel(**{"PASS_NUM": p}),
                kdims=["X", "Y"],
                vdims=["TIC"],
            ).opts(
                colorbar=False,
                cmap="viridis",
                data_aspect=1,
                xlabel="x [μm]",
                ylabel="y [μm]",
                title=f"Pass {int(p)}",
            )
            for p in _pass_vals
        },
        kdims=["PASS_NUM"],
    ).cols(3)

    hv_mpl_svg_render(_tic_grid)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Decay Curves of 3 laser energy settings
    """)
    return


@app.cell
def _():
    hv.extension("matplotlib")

    _base = Path(
        "/mnt/eternus/users/Davide/maldi-deconv-rl/data/decay_curve/SMALDI 5/Cerebellum"
    )

    _settings = {
        "A42": _base / "DHAP_positive_A42" / "Decay_Profile.csv",
        "A43.5": _base / "DHAP_positive_A43.5" / "Decay_Profile.csv",
        "A45": _base / "DHAP_positive_A45" / "Decay_Profile.csv",
    }

    _curves = []
    for _label, _path in _settings.items():
        _df = pl.read_csv(_path)
        if "pass_num" in _df.columns:
            _df = _df.rename({"pass_num": "PASS_NUM"})
        _df = _df.with_columns(pl.lit(_label).alias("Attenuator"))
        _curves.append(_df)

    _all = pl.concat(_curves)

    decay_overlay = decay_curve_group_overlay(
        _all,
        group_col="Attenuator",
        x_col="PASS_NUM",
        y_col="value",
        yerr_col="error",
        # colors={"A42": "#1f77b4", "A43.5": "#ff7f0e", "A45": "#2ca02c"},
        # markers={"A42": "o", "A43.5": "s", "A45": "^"},
        title="Decay Curves – 3 Laser Energy Settings (Cerebellum)",
    )

    hv_mpl_svg_render(decay_overlay)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Decay curves from cerebellum and kidney with the same laser energy setting
    """)
    return


@app.cell
def _():
    _base = Path(
        "/mnt/eternus/users/Davide/maldi-deconv-rl/data/decay_curve/SMALDI 5"
    )

    _tissues = {
        "Cerebellum": _base
        / "Cerebellum"
        / "DHAP_positive_A43.5"
        / "Decay_Profile.csv",
        "Kidney": _base / "Kidney" / "DHAP_pos_kidney" / "Decay_Profile.csv",
    }

    _curves2 = []
    for _label, _path in _tissues.items():
        _df = pl.read_csv(_path)
        if "pass_num" in _df.columns:
            _df = _df.rename({"pass_num": "PASS_NUM"})
        _df = _df.with_columns(pl.lit(_label).alias("Tissue Type"))
        _curves2.append(_df)

    _all2 = pl.concat(_curves2)

    tissue_overlay = decay_curve_group_overlay(
        _all2,
        group_col="Tissue Type",
        x_col="PASS_NUM",
        y_col="value",
        yerr_col="error",
        colors={"Cerebellum": "#1f77b4", "Kidney": "#d62728"},
        markers={"Cerebellum": "o", "Kidney": "s"},
        title="Decay Curves – Cerebellum vs Kidney (DHAP, A43.5, Positive-Ion Mode)",
    )

    hv_mpl_svg_render(tissue_overlay)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Decay curve of negative mode
    """)
    return


@app.cell
def _():
    _base = Path(
        "/mnt/eternus/users/Davide/maldi-deconv-rl/data/decay_curve/SMALDI 5/Cerebellum"
    )

    _neg_settings = {
        "A42": _base / "DHAP_NEG_A42" / "Decay_Profile.csv",
        "A43.5": _base / "DHAP_NEG_A43.5" / "Decay_Profile.csv",
    }

    _curves_neg = []
    for _label, _path in _neg_settings.items():
        _df = pl.read_csv(_path)
        if "pass_num" in _df.columns:
            _df = _df.rename({"pass_num": "PASS_NUM"})
        _df = _df.with_columns(pl.lit(_label).alias("Attenuator"))
        _curves_neg.append(_df)

    _all_neg = pl.concat(_curves_neg)

    neg_overlay = decay_curve_group_overlay(
        _all_neg,
        group_col="Attenuator",
        x_col="PASS_NUM",
        y_col="value",
        yerr_col="error",
        colors={"A42": "#1f77b4", "A43.5": "#ff7f0e"},
        markers={"A42": "o", "A43.5": "s"},
        title="Decay Curves – Negative Mode Cerebellum",
    )

    hv_mpl_svg_render(neg_overlay)
    return


@app.cell
def _():
    _base = Path(
        "/mnt/eternus/users/Davide/maldi-deconv-rl/data/decay_curve/SMALDI 5/Cerebellum"
    )

    _settings = {
        "DHB_A43.5": _base / "DHB_positive_A43.5" / "Decay_Profile.csv",
        "CHCA_A42": _base / "CHCA_positive_A42" / "Decay_Profile.csv",
        "DHAP_A43.5": _base / "DHAP_positive_A43.5" / "Decay_Profile.csv",
    }

    _curves = []
    for _label, _path in _settings.items():
        _df = pl.read_csv(_path)
        if "pass_num" in _df.columns:
            _df = _df.rename({"pass_num": "PASS_NUM"})
        _df = _df.with_columns(pl.lit(_label).alias("Matrix Type"))
        _curves.append(_df)

    _all = pl.concat(_curves)

    matrix_overlay = decay_curve_group_overlay(
        _all,
        group_col="Matrix Type",
        x_col="PASS_NUM",
        y_col="value",
        yerr_col="error",
        # colors={"A42": "#1f77b4", "A43.5": "#ff7f0e", "A45": "#2ca02c"},
        # markers={"A42": "o", "A43.5": "s", "A45": "^"},
        title="Decay Curves – Matrix (Cerebellum, Positive-Ion Mode)",
    )

    hv_mpl_svg_render(matrix_overlay)
    return (matrix_overlay,)


@app.cell
def _(matrix_overlay):
    hv_mpl_svg_render(hv.Layout(matrix_overlay.values()).cols(3))
    return


if __name__ == "__main__":
    app.run()
