import marimo

__generated_with = "0.23.10"
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
    from datashader.transfer_functions import shade, stack

    from PIL import Image, ImageChops

    from pyimzml.ImzMLParser import ImzMLParser

    import sys
    from pathlib import Path as _Path

    _root_dir = _Path(__file__).resolve().parent.parent
    if str(_root_dir) not in sys.path:
        sys.path.append(str(_root_dir))

    from src.imzml_io import load_dataarray
    from src.plotting import set_hv_backend, decay_curve_group_overlay

    set_hv_backend("matplotlib")


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
        "data/decay_curve/SMALDI 5/Cerebellum/DHAP_positive_A43.5/multiple_passes_filtered.h5"
    )

    data_xr = load_dataarray(manual_file).sortby('M_Z')
    intensity_ratios_xr = data_xr / data_xr.sel(M_Z=760.5851, method='nearest')
    return (intensity_ratios_xr,)


@app.cell
def _(intensity_ratios_xr):
    hv.output(backend='bokeh')

    pass_nums = [1, 3, 5, 9]
    m_z_vals = intensity_ratios_xr.M_Z.values

    grid = hv.GridSpace(kdims=['m/z', 'Pass'])

    for p in pass_nums:
        for mz in m_z_vals:
            img_data = intensity_ratios_xr.sel(PASS_NUM=p, M_Z=mz).values

            img = hv.Image(img_data, kdims=['x', 'y']).opts(colorbar=False, cmap="viridis")

            grid[mz, p] = img

    grid.opts(
        shared_xaxis=True, 
        shared_yaxis=True,
        plot_size=200,
        axiswise=True,
    )

    grid
    return


@app.cell
def _():
    hv.help(hv.Image.opts)
    return


@app.cell
def _():
    # _slice = intensity_ratios_xr.sel(
    #     M_Z=mz_slider.value, ITERATION=iter_slider.value
    # )

    # intensity_ratios_xr.sel(PASS_NUM=[1,3,5,9]).hvplot.image(
    #     x="X",
    #     y="Y",
    #     colorbar=True,
    #     cmap="viridis",
    #     data_aspect=1,
    #     title=f"m/z={mz_slider.value:.4f}  iteration={iter_slider.value}",
    # )
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

    hv.NdLayout(
        {
            int(p): hv.Image(
                tic_passes_xr.sel(**{"PASS_NUM": p}),
                kdims=["X", "Y"],
                vdims=["TIC"],
            ).opts(colorbar=False, cmap="viridis", data_aspect=1, width=200)
            for p in _pass_vals
        },
        kdims=["PASS_NUM"],
    ).cols(3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Decay Curves of 3 laser energy settings
    """)
    return


@app.cell
def _():
    set_hv_backend("matplotlib", fig_format=None)

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
        _df = pl.read_csv(_path).with_columns(pl.lit(_label).alias("setting"))
        _curves.append(_df)

    _all = pl.concat(_curves).to_pandas()

    decay_overlay = decay_curve_group_overlay(
        _all,
        group_col="setting",
        x_col="PASS_NUM",
        y_col="value",
        yerr_col="error",
        colors={"A42": "#1f77b4", "A43.5": "#ff7f0e", "A45": "#2ca02c"},
        markers={"A42": "o", "A43.5": "s", "A45": "^"},
        title="Decay Curves – 3 Laser Energy Settings",
    )

    decay_overlay
    return (decay_overlay,)


@app.cell
def _(decay_overlay):
    hv.save(decay_overlay, r"data/figures/fig2/Panel_D_decay_curves_cerebellum.svg" )
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
        _df = pl.read_csv(_path).with_columns(pl.lit(_label).alias("tissue"))
        _curves2.append(_df)

    _all2 = pl.concat(_curves2).to_pandas()

    tissue_overlay = decay_curve_group_overlay(
        _all2,
        group_col="tissue",
        x_col="PASS_NUM",
        y_col="value",
        yerr_col="error",
        colors={"Cerebellum": "#1f77b4", "Kidney": "#d62728"},
        markers={"Cerebellum": "o", "Kidney": "s"},
        title="Decay Curves – Cerebellum vs Kidney (A43.5)",
    )

    tissue_overlay
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
        _df = pl.read_csv(_path).with_columns(pl.lit(_label).alias("setting"))
        _curves_neg.append(_df)

    _all_neg = pl.concat(_curves_neg).to_pandas()

    neg_overlay = decay_curve_group_overlay(
        _all_neg,
        group_col="setting",
        x_col="PASS_NUM",
        y_col="value",
        yerr_col="error",
        colors={"A42": "#1f77b4", "A43.5": "#ff7f0e"},
        markers={"A42": "o", "A43.5": "s"},
        title="Decay Curves – Negative Mode (Cerebellum)",
    )

    neg_overlay
    return


if __name__ == "__main__":
    app.run()
