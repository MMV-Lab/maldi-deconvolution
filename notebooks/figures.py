import marimo

__generated_with = "0.23.5"
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
    from matplotlib.colors import LinearSegmentedColormap

    hv.extension("matplotlib")
    hv.output(backend="matplotlib", fig="svg")

    from PIL import Image, ImageChops

    from pyimzml.ImzMLParser import ImzMLParser


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

    data_xr = xr.open_dataarray(manual_file).sortby('m_z')
    intensity_ratios_xr = data_xr / data_xr.sel(m_z=760.5851, method='nearest')
    return (intensity_ratios_xr,)


@app.cell
def _(intensity_ratios_xr):
    hv.output(backend='bokeh')

    pass_nums = [1, 3, 5, 9]
    m_z_vals = intensity_ratios_xr.m_z.values

    grid = hv.GridSpace(kdims=['m/z', 'Pass'])

    for p in pass_nums:
        for mz in m_z_vals:
            img_data = intensity_ratios_xr.sel(pass_num=p, m_z=mz).values

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
    #     m_z=mz_slider.value, iteration=iter_slider.value
    # )

    # intensity_ratios_xr.sel(pass_num=[1,3,5,9]).hvplot.image(
    #     x="x",
    #     y="y",
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
            "pass_num": np.arange(1, _n_passes + 1),
            "y": np.arange(1, _max_y + 1),
            "x": np.arange(1, _max_x + 1),
        },
        dims=["pass_num", "y", "x"],
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
    _pass_vals = sorted(tic_passes_xr.coords["pass_num"].values.tolist())

    hv.NdLayout(
        {
            int(p): hv.Image(
                tic_passes_xr.sel(**{"pass_num": p}),
                kdims=["x", "y"],
                vdims=["TIC"],
            ).opts(colorbar=False, cmap="viridis", data_aspect=1, width=200)
            for p in _pass_vals
        },
        kdims=["pass_num"],
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

    _colors = {"A42": "#1f77b4", "A43.5": "#ff7f0e", "A45": "#2ca02c"}
    _markers = {"A42": "o", "A43.5": "s", "A45": "^"}

    decay_overlay = hv.NdOverlay(
        {
            _label: (
                hv.ErrorBars(
                    _all[_all["setting"] == _label],
                    kdims=["pass_num"],
                    vdims=["value", "error"],
                ).opts(color=_colors[_label], alpha=0.4)
                * hv.Curve(
                    _all[_all["setting"] == _label],
                    kdims=["pass_num"],
                    vdims=["value"],
                    label=_label,
                ).opts(color=_colors[_label])
                * hv.Scatter(
                    _all[_all["setting"] == _label],
                    kdims=["pass_num"],
                    vdims=["value"],
                ).opts(color=_colors[_label], marker=_markers[_label], s=50)
            )
            for _label in _settings
        }
    ).opts(
        hv.opts.Curve(xlabel="Pass", ylabel="Normalised Intensity"),
        hv.opts.NdOverlay(
            title="Decay Curves – 3 Laser Energy Settings",
        ),
    )

    decay_overlay
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

    _colors2 = {"Cerebellum": "#1f77b4", "Kidney": "#d62728"}
    _markers2 = {"Cerebellum": "o", "Kidney": "s"}

    tissue_overlay = hv.NdOverlay(
        {
            _label: (
                hv.ErrorBars(
                    _all2[_all2["tissue"] == _label],
                    kdims=["pass_num"],
                    vdims=["value", "error"],
                ).opts(color=_colors2[_label], alpha=0.4)
                * hv.Curve(
                    _all2[_all2["tissue"] == _label],
                    kdims=["pass_num"],
                    vdims=["value"],
                    label=_label,
                ).opts(color=_colors2[_label])
                * hv.Scatter(
                    _all2[_all2["tissue"] == _label],
                    kdims=["pass_num"],
                    vdims=["value"],
                ).opts(color=_colors2[_label], marker=_markers2[_label], s=50)
            )
            for _label in _tissues
        }
    ).opts(
        hv.opts.Curve(xlabel="Pass", ylabel="Normalised Intensity"),
        hv.opts.NdOverlay(title="Decay Curves – Cerebellum vs Kidney (A43.5)"),
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

    _colors_neg = {"A42": "#1f77b4", "A43.5": "#ff7f0e"}
    _markers_neg = {"A42": "o", "A43.5": "s"}

    neg_overlay = hv.NdOverlay(
        {
            _label: (
                hv.ErrorBars(
                    _all_neg[_all_neg["setting"] == _label],
                    kdims=["pass_num"],
                    vdims=["value", "error"],
                ).opts(color=_colors_neg[_label], alpha=0.4)
                * hv.Curve(
                    _all_neg[_all_neg["setting"] == _label],
                    kdims=["pass_num"],
                    vdims=["value"],
                    label=_label,
                ).opts(color=_colors_neg[_label])
                * hv.Scatter(
                    _all_neg[_all_neg["setting"] == _label],
                    kdims=["pass_num"],
                    vdims=["value"],
                ).opts(
                    color=_colors_neg[_label], marker=_markers_neg[_label], s=50
                )
            )
            for _label in _neg_settings
        }
    ).opts(
        hv.opts.Curve(xlabel="Pass", ylabel="Normalised Intensity"),
        hv.opts.NdOverlay(title="Decay Curves – Negative Mode (Cerebellum)"),
    )

    neg_overlay
    return


if __name__ == "__main__":
    app.run()
