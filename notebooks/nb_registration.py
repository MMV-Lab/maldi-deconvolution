import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    import polars as pl
    import numpy as np
    import xarray as xr

    from pathlib import Path
    from bioio import BioImage
    import bioio_bioformats

    import holoviews as hv
    import hvplot
    import hvplot.xarray
    import hvplot.polars

    import SimpleITK as sitk

    from sklearn.metrics import (
        auc,
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
    from scipy.ndimage import distance_transform_edt

    from sklearn.metrics import normalized_mutual_info_score

    from scipy.ndimage import find_objects, label

    import sys

    _root_dir = Path(__file__).resolve().parent.parent
    if str(_root_dir) not in sys.path:
        sys.path.append(str(_root_dir))

    from src.registration import (
        rgb_to_grayscale,
        mask_from_threshold,
        mask_from_polygons,
        smooth_closed_contour,
        build_sitk_image,
        register_images,
        apply_transform_resample,
        crop_to_transformed_bbox,
    )
    from src.registration_metrics import (
        cohens_d,
        inside_outside_ratio,
        digitize_for_nmi,
        best_threshold_from_pr,
        compute_boundary_metrics,
        radial_intensity_profile,
        edge_rise_width,
        lsf_fwhm,
    )
    from src.visualization import build_colormap_dict
    from src.imzml_io import (
        DATAARRAY_EXTENSIONS,
        load_dataarray,
        pixel_sizes,
        save_dataarray,
    )
    hv.extension("bokeh")


@app.cell
def _():
    ref_select_widget = mo.ui.file_browser(
        multiple=False,
        selection_mode="file",
        filetypes=[".tif"],
    )
    maldi_select_widget = mo.ui.file_browser(
        multiple=False,
        selection_mode="all",
        filetypes=DATAARRAY_EXTENSIONS,
    )

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(
                        "# Select the **tif** file containing the **reference microscopy image** (H&E/RGB or fluorescence):"
                    ),
                    ref_select_widget,
                ]
            ),
            mo.vstack(
                [
                    mo.md(
                        "# Select the **ome.zarr** store containing the **deconvolved MALDI** data:"
                    ),
                    maldi_select_widget,
                ]
            ),
        ]
    )
    return maldi_select_widget, ref_select_widget


@app.cell
def _(maldi_select_widget, ref_select_widget):
    REF_image_path = ref_select_widget.value[0].path
    REF_image = BioImage(REF_image_path)

    try:
        _x_size = REF_image.physical_pixel_sizes.X
        _y_size = REF_image.physical_pixel_sizes.Y

        if _x_size is not None and _y_size is not None:
            if _x_size == _y_size:
                _tmp_pixel_size = str(_x_size)
            else:
                # If both exist but are different, take the average
                _tmp_pixel_size = str((_x_size + _y_size) / 2)
        else:
            # Fallback if sizes are None
            _tmp_pixel_size = str(_x_size or _y_size or "")
    except (AttributeError, TypeError):
        # Fallback if physical_pixel_sizes doesn't exist or isn't accessible
        _tmp_pixel_size = ""

    REF_pixel_size_text = mo.ui.text(
        value=_tmp_pixel_size if _tmp_pixel_size else None
    )

    deconv_image_path = maldi_select_widget.value[0].path
    deconv_image = load_dataarray(deconv_image_path, lazy=False)

    _mz = deconv_image.M_Z.values.tolist()
    _iter = deconv_image.ITERATION.values.tolist()

    # Modality detection
    REF_image_raw = REF_image.xarray_data.squeeze(drop=True)
    is_rgb = "S" in REF_image_raw.dims
    is_fluorescence = "C" in REF_image_raw.dims

    # Channel selector for fluorescence
    if is_fluorescence:
        _channels = REF_image_raw.coords["C"].values.tolist()
        REF_channel_widget = mo.ui.dropdown(
            options={str(ch): ch for ch in _channels},
            value=str(_channels[0]),
            label="Fluorescence Channel:",
        )
    else:
        REF_channel_widget = None

    MALDI_m_z_registration_widget = mo.ui.multiselect(
        _mz, max_selections=1, label="m/z:"
    )
    MALDI_iteration_registration_widget = mo.ui.multiselect(
        _iter,
        max_selections=1,
        label="iteration:",
        value=["6"],
    )

    MALDI_m_z_transform_widget = mo.ui.multiselect(_mz, label="m/z:")
    MALDI_iteration_transform_widget = mo.ui.multiselect(
        _iter,
        label="iteration:",
        value=["0"],
    )

    # Build UI elements conditionally
    _ref_ui_elements = [
        mo.md(f"Pixel Size Reference Image: {REF_pixel_size_text} μm / pixel"),
    ]
    if is_fluorescence and REF_channel_widget is not None:
        _ref_ui_elements.append(REF_channel_widget)
        _ref_ui_elements.append(
            mo.md(
                f"**Detected: Fluorescence image** ({len(_channels)} channels)"
            )
        )
    elif is_rgb:
        _ref_ui_elements.append(mo.md("**Detected: RGB/H&E image**"))

    mo.hstack(
        [
            mo.vstack(_ref_ui_elements),
            mo.vstack(
                [
                    mo.md(
                        f"""Select the values of m/z and iteration used **during registration**"""
                    ),
                    MALDI_m_z_registration_widget,
                    MALDI_iteration_registration_widget,
                    mo.md(
                        "Select the values you also want to transform **after registration**"
                    ),
                    MALDI_m_z_transform_widget,
                    MALDI_iteration_transform_widget,
                ]
            ),
        ]
    )
    return (
        MALDI_iteration_registration_widget,
        MALDI_iteration_transform_widget,
        MALDI_m_z_registration_widget,
        MALDI_m_z_transform_widget,
        REF_channel_widget,
        REF_image_raw,
        REF_pixel_size_text,
        deconv_image,
        deconv_image_path,
        is_fluorescence,
        is_rgb,
    )


@app.cell
def _(
    REF_channel_widget,
    REF_image_raw,
    REF_pixel_size_text,
    is_fluorescence,
    is_rgb,
):
    REF_pixel_um = float(REF_pixel_size_text.value)

    REF_image_xr = REF_image_raw.copy()

    REF_image_xr = REF_image_xr.astype(REF_image_raw.dtype.newbyteorder("<"), casting="equiv")

    # Process based on modality
    if is_fluorescence:
        # Fluorescence: select channel and remove C dimension
        selected_channel = REF_channel_widget.value
        REF_image_xr = REF_image_xr.sel(C=selected_channel)
        REF_image_xr.attrs["modality"] = "fluorescence"
        REF_image_xr.attrs["channel"] = str(selected_channel)
    elif is_rgb:
        # RGB: rename S dimension if it exists
        if "S" not in REF_image_xr.dims and "C" in REF_image_xr.dims:
            REF_image_xr = REF_image_xr.rename({"C": "S"})
        REF_image_xr.attrs["modality"] = "rgb"
    else:
        # Fallback: treat as grayscale
        REF_image_xr.attrs["modality"] = "grayscale"

    # Assign physical coordinates
    _coord_x = np.arange(REF_image_xr.sizes["X"]) * REF_pixel_um
    _coord_y = np.arange(REF_image_xr.sizes["Y"]) * REF_pixel_um

    REF_image_xr = REF_image_xr.assign_coords(X=_coord_x, Y=_coord_y)

    # Add S coordinate for RGB if needed
    if "S" in REF_image_xr.dims:
        REF_image_xr = REF_image_xr.assign_coords(S=["R", "G", "B"])

    REF_image_xr.coords["X"].attrs["units"] = "μm"
    REF_image_xr.coords["Y"].attrs["units"] = "μm"

    # Display based on modality
    if is_rgb and "S" in REF_image_xr.dims:
        _plot = REF_image_xr.hvplot.rgb(
            x="X",
            y="Y",
            bands="S",
            data_aspect=1,
            flip_yaxis=True,
            rasterize=True,
        )
    else:
        # Fluorescence or grayscale
        _plot = REF_image_xr.hvplot.image(
            x="X",
            y="Y",
            data_aspect=1,
            flip_yaxis=True,
            rasterize=True,
            cmap="gray",
        )

    _plot
    return REF_image_xr, REF_pixel_um


@app.cell
def _(
    MALDI_iteration_registration_widget,
    MALDI_iteration_transform_widget,
    MALDI_m_z_registration_widget,
    MALDI_m_z_transform_widget,
    deconv_image,
):
    # Extract values from widgets
    MALDI_iteration_registration = MALDI_iteration_registration_widget.value[0]
    MALDI_m_z_registration = MALDI_m_z_registration_widget.value[0]

    MALDI_iterations_transform = MALDI_iteration_transform_widget.value
    MALDI_m_z_transform = MALDI_m_z_transform_widget.value

    # Create xarray for registration (used during landmark-based registration)
    MALDI_image_registration_xr = deconv_image.sel(
        ITERATION=MALDI_iteration_registration,
        M_Z=MALDI_m_z_registration,
    )

    MALDI_image_registration_xr.hvplot.image(
        x="X",
        y="Y",
        data_aspect=1,
        flip_yaxis=True,
        clim=(0, MALDI_image_registration_xr.quantile(0.98).item()),
        cmap="grey",
    )
    return (
        MALDI_image_registration_xr,
        MALDI_iterations_transform,
        MALDI_m_z_registration,
        MALDI_m_z_transform,
    )


@app.cell(hide_code=True)
def _():
    # img_plot = HE_image_xr.hvplot.rgb(
    #     x="X",
    #     y="Y",
    #     bands="S",
    #     data_aspect=1,
    #     flip_yaxis=True,
    #     rasterize=True,
    #     title="H&E",
    #     width=650,
    #     height=550,
    #     hover=False,
    # )

    # points_layer = hv.Points([], vdims=[]).opts(
    #     color="red",
    #     size=10,
    # )

    # # 3. Create the PointDraw stream and link it to the points layer
    # point_stream = hv.streams.PointDraw(source=points_layer, empty_value="black")

    # # 4. Overlay the points on top of your image
    # interactive_plot = (img_plot * points_layer).opts(
    #     active_tools=["pan", "wheel_zoom"]
    # )

    # # Display the combined interactive plot
    # interactive_plot
    return


@app.cell(hide_code=True)
def _():
    # img_plot = HE_image_xr.hvplot.rgb(
    #     x="X",
    #     y="Y",
    #     bands="S",
    #     data_aspect=1,
    #     flip_yaxis=True,
    #     rasterize=True,
    #     title="H&E",
    #     width=650,
    #     height=550,
    #     hover=False,
    # )

    # points_layer = hv.Points([], vdims=[]).opts(
    #     color="red",
    #     size=10,
    # )

    # # 3. Create the PointDraw stream and link it to the points layer
    # point_stream = hv.streams.PointDraw(source=points_layer, empty_value="black")

    # # 4. Overlay the points on top of your image
    # interactive_plot = (img_plot * points_layer).opts(active_tools=["pan", "wheel_zoom"])

    # # Display the combined interactive plot
    # interactive_plot
    return


@app.cell
def _(MALDI_image_registration_xr, REF_image_xr, is_rgb):
    hv.extension("bokeh")

    _vmax = float(MALDI_image_registration_xr.quantile(0.98))


    # Invisible-until-drawn Points elements — required by PointDraw as Bokeh renderer source
    _REF_pts = hv.Points([]).opts(
        size=10,
        color="green",
        marker="x",
        line_width=2.5,
    )

    _MALDI_pts = hv.Points([]).opts(
        size=10,
        color="cyan",
        marker="x",
        line_width=2.5,
    )

    REF_lm_stream = hv.streams.PointDraw(
        source=_REF_pts, data={"x": [], "y": []}
    )
    MALDI_lm_stream = hv.streams.PointDraw(
        source=_MALDI_pts, data={"x": [], "y": []}
    )

    # Build reference image plot based on modality
    if is_rgb and "S" in REF_image_xr.dims:
        _REF_base = REF_image_xr.hvplot.rgb(
            x="X",
            y="Y",
            bands="S",
            data_aspect=1,
            flip_yaxis=True,
            rasterize=True,
            title="Reference Image (RGB)",
            width=650,
            height=550,
            hover=False,
        )
    else:
        _REF_vmax = float(REF_image_xr.quantile(0.99))
        _REF_base = REF_image_xr.hvplot.image(
            x="X",
            y="Y",
            data_aspect=1,
            flip_yaxis=True,
            clim=(0, _REF_vmax),
            rasterize=True,
            cmap="viridis",
            title="Reference Image (Fluorescence)",
            width=650,
            height=550,
            hover=False,
            # robust=True,
            colorbar=False,
        )

    _REF_combined = (_REF_base * _REF_pts).opts(
        active_tools=["pan", "wheel_zoom"]
    )

    _MALDI_combined = (
        MALDI_image_registration_xr.hvplot.image(
            x="X",
            y="Y",
            data_aspect=1,
            flip_yaxis=True,
            clim=(0, _vmax),
            cmap="viridis",
            title="MALDI",
            width=650,
            height=550,
            hover=False,
            colorbar=False,
        )
        * _MALDI_pts
    ).opts(active_tools=["pan", "wheel_zoom"])

    landmark_selection_plot = (_REF_combined + _MALDI_combined).opts(
        shared_axes=False, framewise=True
    )
    landmark_selection_plot
    return MALDI_lm_stream, REF_lm_stream


@app.cell
def _():
    # fig 2
    # HE_lm_stream.event(
    #     data={
    #         "x": [
    #             196.30297077007592,
    #             187.07009228494155,
    #             123.9970418903857,
    #             116.03573151896603,
    #             105.50543815008173,
    #         ],
    #         "y": [
    #             291.95073462178357,
    #             294.36885591928785,
    #             265.9870712268268,
    #             267.9809538709276,
    #             209.453894104174,
    #         ],
    #     }
    # )

    # MALDI_lm_stream.event(
    #     data={
    #         "x": [
    #             196.66396064088806,
    #             188.28952298323952,
    #             125.7931169091972,
    #             118.05747183320854,
    #             107.5319092870344,
    #         ],
    #         "y": [
    #             190.52673580847897,
    #             192.7684070407323,
    #             167.72249060642847,
    #             169.37183862166697,
    #             110.5842605338397,
    #         ],
    #     }
    # )
    return


@app.cell
def _():
    # fig.3ab

    # HE_lm_stream.event(
    #     data={
    #         "x": [
    #             186.55354860286582,
    #             167.14481424033215,
    #             272.4225503202458,
    #             273.59881957454047,
    #             246.54422287960236,
    #         ],
    #         "y": [
    #             148.64070684901603,
    #             241.4387579838232,
    #             49.17091098942782,
    #             148.64070684901603,
    #             228.09522226022173,
    #         ],
    #     }
    # )

    # MALDI_lm_stream.event(
    #     data={
    #         "x": [
    #             114.93448868568497,
    #             97.05483373776829,
    #             197.08398363517992,
    #             195.15101010101012,
    #             172.43910565617108,
    #         ],
    #         "y": [
    #             176.1723874727885,
    #             253.4140351295471,
    #             62.05400851567586,
    #             164.71068658828736,
    #             258.8956941922506,
    #         ],
    #     }
    # )
    return


@app.cell
def _():
    # # fig 3c

    # REF_lm_stream.event(data={
    #     "x": [
    #       92.52700933839823,
    #       174.10555971035134,
    #       174.3757038751714,
    #       154.92649872718155,
    #       117.91900484396432
    #     ],
    #     "y": [
    #       243.6995303952372,
    #       250.94231387194804,
    #       192.44277807918587,
    #       175.45006311487262,
    #       177.12147386899278
    #     ]
    #   }
    # )

    # MALDI_lm_stream.event(data=
    #   {
    #     "x": [
    #       42.18253370946747,
    #       114.58544146833356,
    #       116.59665395150324,
    #       98.49589504475038,
    #       66.31685334484219
    #     ],
    #     "y": [
    #       218.24958870670056,
    #       233.11355212376424,
    #       176.76875320620525,
    #       166.74421268569122,
    #       164.67017618522743
    #     ]
    #   }
    # )
    return


@app.cell
def _():
    # # fucci fig 3

    # REF_lm_stream.event(data=
    #   {
    #     "x": [
    #       107.7567620025882,
    #       100.76429197900602,
    #       126.23682596658055,
    #       102.26265391598334,
    #       198.65875799146863,
    #       147.71369001631956,
    #       138.72344218266363,
    #       223.13238402105827
    #     ],
    #     "y": [
    #       1033.8688376856246,
    #       996.1450969138434,
    #       1008.8897721201462,
    #       1068.534603166206,
    #       985.4390719016737,
    #       939.0485786104502,
    #       913.6867994096265,
    #       1002.2622920934311
    #     ]
    #   }
    # )

    # MALDI_lm_stream.event(data={
    #     "x": [
    #       64.72998639809967,
    #       59.69391720241973,
    #       83.95883016412911,
    #       59.69391720241973,
    #       154.4643577760639,
    #       108.22367326678655,
    #       101.81410525081137,
    #       182.84969733960705
    #     ],
    #     "y": [
    #       166.04315909067785,
    #       126.73811618804928,
    #       141.37419883728023,
    #       193.78111481984453,
    #       119.65608493804928,
    #       82.35727175394689,
    #       46.9471155039469,
    #       140.42958220163976
    #     ]
    #   }
    # )
    return


@app.cell
def _(MALDI_lm_stream, REF_lm_stream):
    # # HOECHST DATASET 

    REF_lm_stream.event(
        data={
            "x": [
                407.15012687894193,
                394.18618225235554,
                385.2455307857443,
                463.92326369192364,
                458.55887281195686,
                536.3425405714752,
                540.8128663047809,
                386.9294737552241,
            ],
            "y": [
                526.4979367633516,
                526.9589391045988,
                533.4129718820589,
                467.4896370837171,
                505.75283140723013,
                361.45909859687384,
                327.8059276858323,
                438.19590710521726,
            ],
        }
    )
    MALDI_lm_stream.event(
        data={
            "x": [
                158.1707695265411,
                148.80166373093067,
                143.40733009103377,
                215.2371411907137,
                214.10149200336696,
                288.77042607141345,
                291.8934613366169,
                143.0760541616171,
            ],
            "y": [
                296.7245176129544,
                298.18844039351853,
                304.04413151577506,
                236.7036836098251,
                267.153277445559,
                126.32390595528994,
                98.80215768068432,
                205.52454928440082,
            ],
        }
    )
    return


@app.cell
def _():
    # A375 - CC296

    # [
    #   {
    #     "x": [
    #       210.09920057812775,
    #       210.09920057812775,
    #       378.6519144743002,
    #       487.96092013710194,
    #       436.2268728297185,
    #       389.49939346197175,
    #       406.18778349767814,
    #       420.3728895635663,
    #       327.7523375976268,
    #       278.5215742600617,
    #       109.96886036388922,
    #       146.6832993440966
    #     ],
    #     "y": [
    #       127.86865956424657,
    #       145.93904376967154,
    #       115.82171487710761,
    #       158.8464704379132,
    #       202.731740805105,
    #       235.43054846626057,
    #       217.36013143555215,
    #       237.1515452537495,
    #       246.61697834701334,
    #       184.66132377439658,
    #       157.12547365042428,
    #       186.3823205618855
    #     ]
    #   },
    #   {
    #     "x": [
    #       237.3166666666667,
    #       240.34606933593753,
    #       403.93000000000006,
    #       515.257976888021,
    #       466.0313102213542,
    #       421.3486897786459,
    #       436.49535644531255,
    #       448.6127360026042,
    #       357.7327360026042,
    #       308.5060693359375,
    #       138.86333333333334,
    #       176.73000000000002
    #     ],
    #     "y": [
    #       139.48447321891786,
    #       157.44746726036072,
    #       121.52147917747497,
    #       164.47647917747497,
    #       208.2124672603607,
    #       241.79546130180358,
    #       224.6134851360321,
    #       244.1384851360321,
    #       255.07246726036072,
    #       196.4974672603607,
    #       167.6004613018036,
    #       197.27848513603212
    #     ]
    #   }
    # ]


    # [
    #   {
    #     "x": [
    #       406.8515037824614,
    #       327.80836641044965,
    #       437.5106455877838,
    #       420.02535848827347,
    #       486.36038196970094,
    #       282.59812149449766,
    #       271.513424000103
    #     ],
    #     "y": [
    #       217.8304705306969,
    #       245.74258272677628,
    #       200.53978622931862,
    #       238.5792954614209,
    #       161.07997557660582,
    #       192.34728737721605,
    #       176.20933075296574
    #     ]
    #   },
    #   {
    #     "x": [
    #       435.2919012005341,
    #       356.7682293609076,
    #       464.81435418943994,
    #       447.4661124488194,
    #       513.782804795097,
    #       315.117654886347,
    #       304.5923077164577
    #     ],
    #     "y": [
    #       223.788330792473,
    #       255.48881878843014,
    #       207.1533909025014,
    #       244.8173569518841,
    #       165.81700104352674,
    #       205.1635537475609,
    #       187.52545194597164
    #     ]
    #   }
    # ]
    return


@app.cell
def _():
    ## H9c2 - CC307-1

    # [
    #   {
    #     "x": [
    #       1766.1461179791393,
    #       1824.3983804979578,
    #       1754.8714482052824,
    #       1655.0103448219231,
    #       1586.0202918134164,
    #       1975.0395431066468,
    #       1981.3729586821592,
    #       1783.1478121940893,
    #       1919.9123925417257
    #     ],
    #     "y": [
    #       1690.1330947802514,
    #       1693.4209955217468,
    #       1629.5805491315177,
    #       1616.9768616843562,
    #       1604.647163352649,
    #       1547.7193682279994,
    #       1555.2919024102544,
    #       1537.2938504342046,
    #       1652.8983058526856
    #     ]
    #   },
    #   {
    #     "x": [
    #       223.61091189281623,
    #       280.20209478781334,
    #       213.6750410132732,
    #       117.77242773830179,
    #       54.26935042899804,
    #       437.5241210377668,
    #       444.5589398043084,
    #       249.8022407937403,
    #       375.26488406372334
    #     ],
    #     "y": [
    #       168.76872655722195,
    #       173.6691580714525,
    #       109.9635738778049,
    #       95.26227933511329,
    #       80.56099328953815,
    #       35.27980769790463,
    #       42.33836646466419,
    #       18.8876369235521,
    #       134.03347484881166
    #     ]
    #   }
    # ]
    return


@app.cell
def _(MALDI_lm_stream, REF_lm_stream):
    (REF_lm_stream.data, MALDI_lm_stream.data)
    return


@app.cell
def _():
    register_btn = mo.ui.run_button(label="Run Registration")
    mo.vstack(
        [
            mo.md(
                "Select **≥ 3 corresponding landmarks** in both images above "
                "(click to add, drag to move, Backspace to delete), then:"
            ),
            register_btn,
        ]
    )
    return (register_btn,)


@app.cell
def _(
    MALDI_image_registration_xr,
    MALDI_lm_stream,
    REF_image_xr,
    REF_lm_stream,
    REF_pixel_um,
    is_rgb,
    register_btn,
):
    mo.stop(not register_btn.value)

    _REF_x = np.array(REF_lm_stream.data.get("x", []), dtype=np.float64)
    _REF_y = np.array(REF_lm_stream.data.get("y", []), dtype=np.float64)
    _MALDI_x = np.array(MALDI_lm_stream.data.get("x", []), dtype=np.float64)
    _MALDI_y = np.array(MALDI_lm_stream.data.get("y", []), dtype=np.float64)

    mo.stop(
        len(_REF_x) < 3 or len(_MALDI_x) < 3 or len(_REF_x) != len(_MALDI_x),
        mo.md(
            f"Need ≥ 3 matching landmark pairs. "
            f"Got Reference = {len(_REF_x)}, MALDI = {len(_MALDI_x)}."
        ).callout(kind="warn"),
    )

    # Reference image → grayscale float32 for registration metric
    if is_rgb and "S" in REF_image_xr.dims:
        # RGB: convert to grayscale using luminosity method
        _REF_gray = rgb_to_grayscale(
            REF_image_xr, method="luminosity"
        ).values.astype(np.float32)
    else:
        # Fluorescence or grayscale: use directly
        _REF_gray = REF_image_xr.values.astype(np.float32)

    REF_sitk = build_sitk_image(_REF_gray, (REF_pixel_um, REF_pixel_um))

    # MALDI 2D array
    _MALDI_pixel_size_y, _MALDI_pixel_size_x = pixel_sizes(
        MALDI_image_registration_xr
    )
    MALDI_sitk = build_sitk_image(
        MALDI_image_registration_xr.values,
        (_MALDI_pixel_size_x, _MALDI_pixel_size_y),
    )

    # Landmark-based affine initialisation (coordinates are already in µm)
    fixed_landmarks = list(zip(_REF_x, _REF_y))
    moving_landmarks = list(zip(_MALDI_x, _MALDI_y))

    mo.md(
        f"Landmark affine transform initialised from **{len(_REF_x)} point pairs**."
    ).callout(kind="success")
    return MALDI_sitk, REF_sitk, fixed_landmarks, moving_landmarks


@app.cell
def _(
    MALDI_sitk,
    REF_sitk,
    deconv_image_path,
    fixed_landmarks,
    is_rgb,
    moving_landmarks,
):
    _trasform_file = deconv_image_path.parent / "transorm.tfm"

    if _trasform_file.is_file():
        REF_MALDI_transform = sitk.ReadTransform(_trasform_file)
        mo.output.append(
            mo.md("""**Read from file**""").callout(kind="success")
        )
    else:
        REF_MALDI_transform, _metric_value, _stop_condition = register_images(
            REF_sitk,
            MALDI_sitk,
            fixed_landmarks,
            moving_landmarks,
            is_multimodal=is_rgb,
        )

        mo.output.append(
            mo.md(f"""
        **Intensity-based registration complete**
        Final metric value: `{_metric_value:.5f}`
        Stopping condition: `{_stop_condition}`
        """).callout(kind="success")
        )

        # Save the transform to file
        sitk.WriteTransform(REF_MALDI_transform, _trasform_file)
    return (REF_MALDI_transform,)


@app.cell
def _(
    MALDI_iterations_transform,
    MALDI_m_z_transform,
    REF_MALDI_transform,
    REF_image_xr,
    REF_sitk,
    deconv_image_path,
):
    # Use the transform widget values instead of hardcoded values
    _deconv_image = load_dataarray(deconv_image_path, lazy=False)

    def register_MALDI(MALDI_raw):
        return apply_transform_resample(
            MALDI_raw, REF_sitk, REF_MALDI_transform
        )

    _n_mz = len(MALDI_m_z_transform)
    _n_iter = len(MALDI_iterations_transform)

    if _n_mz > 0 and _n_iter > 0:
        # Get Y, X dimensions from first registration
        _first_maldi = _deconv_image.sel(
            ITERATION=MALDI_iterations_transform[0], M_Z=MALDI_m_z_transform[0]
        )
        _first_registered = register_MALDI(_first_maldi)
        _y_size, _x_size = _first_registered.shape

        # Preallocate 4D array
        _all_arrays = np.zeros(
            (_n_mz, _n_iter, _y_size, _x_size), dtype=np.float32
        )

        # Fill array with registered data
        for i_mz, _mz in enumerate(MALDI_m_z_transform):
            for i_iter, _iter in enumerate(MALDI_iterations_transform):
                if i_mz == 0 and i_iter == 0:
                    # We already registered the first one
                    _all_arrays[i_mz, i_iter, :, :] = _first_registered
                else:
                    _MALDI_raw = _deconv_image.sel(ITERATION=_iter, M_Z=_mz)
                    _registered_arr = register_MALDI(_MALDI_raw)
                    _all_arrays[i_mz, i_iter, :, :] = _registered_arr

        MALDI_image_transformed_xr = xr.DataArray(
            _all_arrays,
            dims=["M_Z", "ITERATION", "Y", "X"],
            coords={
                "M_Z": MALDI_m_z_transform,
                "ITERATION": MALDI_iterations_transform,
                "Y": REF_image_xr.coords["Y"],
                "X": REF_image_xr.coords["X"],
            },
        )
    else:
        MALDI_image_transformed_xr = None

    # --- Crop both REF and MALDI images to the transformed MALDI bounding box ---

    if MALDI_image_transformed_xr is not None:
        x_slice, y_slice = crop_to_transformed_bbox(
            REF_MALDI_transform, _deconv_image, REF_image_xr
        )

        # Crop using standard xarray slicing (WITHOUT method="nearest")
        REF_image_transformed_xr = REF_image_xr.sel(Y=y_slice, X=x_slice)

        MALDI_image_transformed_xr = MALDI_image_transformed_xr.sel(
            Y=y_slice, X=x_slice
        )
    return (
        MALDI_image_transformed_xr,
        REF_image_transformed_xr,
        x_slice,
        y_slice,
    )


@app.cell
def _(
    MALDI_image_transformed_xr,
    MALDI_m_z_registration,
    REF_image_raw,
    REF_image_xr,
    deconv_image_path,
    is_fluorescence,
    is_rgb,
    x_slice,
    y_slice,
):
    if MALDI_image_transformed_xr is not None:
        save_dataarray(
            MALDI_image_transformed_xr,
            deconv_image_path.parent
            / f"{deconv_image_path.name}_deconvolved_registered_m_z-{MALDI_m_z_registration}.ome.zarr",
        )

        _output = REF_image_raw.copy()

        _output.attrs.clear()

        _output = _output.assign_coords(
            X=REF_image_xr.X.values, Y=REF_image_xr.Y.values
        )

        if is_fluorescence:
            _output = _output.rename({"C": "M_Z"})
        if is_rgb:
            _output = _output.rename({"S": "M_Z"})

        _output = _output.assign_coords(M_Z=np.arange(len(_output.M_Z)))

        _output = _output.sel(X=x_slice, Y=y_slice)

        save_dataarray(
            _output,
            deconv_image_path.parent
            / f"{deconv_image_path.name}_REFERENCE_registered.ome.zarr",
        )
    return


@app.cell
def _():
    # MALDI_image_transformed_xr.hvplot.image(x="X", y="Y", rasterize=True)
    return


@app.cell
def _(
    MALDI_image_transformed_xr,
    MALDI_m_z_registration,
    REF_image_transformed_xr,
    is_rgb,
):
    hv.extension("bokeh")

    cmap_dict = build_colormap_dict(colormap_names=[])

    if MALDI_m_z_registration in MALDI_image_transformed_xr.M_Z.values:
        _MALDI_for_overlay = MALDI_image_transformed_xr.sel(
            M_Z=MALDI_m_z_registration
        )
    else:
        # Fallback to first m/z if registration m/z not in transform selection
        _MALDI_for_overlay = MALDI_image_transformed_xr.isel(M_Z=0)

    # Select last iteration to show final deconvolved result
    _MALDI_for_overlay = _MALDI_for_overlay.isel(ITERATION=-1)

    _vmax = float(_MALDI_for_overlay.quantile(0.99))

    # Build overlays based on modality using cropped images
    if is_rgb and "S" in REF_image_transformed_xr.dims:
        # RGB path
        _REF_rgb = REF_image_transformed_xr.hvplot.rgb(
            x="X",
            y="Y",
            bands="S",
            data_aspect=1,
            flip_yaxis=True,
            rasterize=True,
            width=900,
            height=700,
            alpha=0.8,
        )

        _REF_gray = rgb_to_grayscale(
            REF_image_transformed_xr, method="mean"
        ).hvplot.image(
            x="X",
            y="Y",
            data_aspect=1,
            flip_yaxis=True,
            rasterize=True,
            cmap=cmap_dict["Linear Magenta"],
            width=900,
            height=700,
            alpha=0.5,
        )
    else:
        # Fluorescence path
        _REF_rgb = REF_image_transformed_xr.hvplot.image(
            x="X",
            y="Y",
            data_aspect=1,
            flip_yaxis=True,
            rasterize=True,
            cmap=cmap_dict["Linear Green"],
            width=900,
            height=700,
            alpha=0.8,
            robust=True,
            colorbar=False,
        )

        _REF_gray = REF_image_transformed_xr.hvplot.image(
            x="X",
            y="Y",
            data_aspect=1,
            flip_yaxis=True,
            rasterize=True,
            cmap=cmap_dict["Linear Magenta"],
            width=900,
            height=700,
            alpha=0.5,
            robust=True,
            colorbar=False,
        )

    _MALDI_hot = _MALDI_for_overlay.clip(0, _vmax).hvplot.image(
        x="X",
        y="Y",
        data_aspect=1,
        flip_yaxis=True,
        rasterize=True,
        cmap="hot",
        clim=(0, _vmax),
        colorbar=False,
        width=900,
        height=700,
        alpha=0.2,
    )

    _MALDI_gray = _MALDI_for_overlay.clip(0, _vmax).hvplot.image(
        x="X",
        y="Y",
        data_aspect=1,
        flip_yaxis=True,
        rasterize=True,
        cmap=cmap_dict["Linear Green"],
        clim=(0, _vmax),
        colorbar=False,
        width=900,
        height=700,
        alpha=0.5,
    )

    (_REF_rgb * _MALDI_hot).opts(
        hv.opts.Overlay(
            title="Registration result: Cropped Reference + MALDI overlay (final iteration)"
        )
    ) + (_REF_gray * _MALDI_gray).opts(
        hv.opts.Overlay(
            title="Registration result: Cropped Reference + MALDI overlay (final iteration)"
        )
    )
    return


@app.cell
def _(REF_image_transformed_xr, REF_image_xr, is_rgb):
    hv.extension("bokeh")

    # 1. Keep your Polygons element as the source of the drawing stream
    _he_polys = hv.Polygons([]).opts(
        fill_alpha=0.0,  # Invisible fill so it doesn't distract
        line_color="green",
    )

    REF_poly_stream = hv.streams.FreehandDraw(source=_he_polys)

    def update_spline(data):
        if not data or "xs" not in data or len(data["xs"]) == 0:
            return hv.Path([])

        paths = []
        for xs, ys in zip(data["xs"], data["ys"]):
            x_smooth, y_smooth = smooth_closed_contour(xs, ys)
            if x_smooth is None:
                continue
            paths.append(hv.Path((x_smooth, y_smooth)))

        return hv.Path(paths).opts(line_color="cyan", line_width=2.5)

    # 3. Wrap the callback function into a DynamicMap bound to your stream
    _smooth_contour_dynamic = hv.DynamicMap(
        update_spline, streams=[REF_poly_stream]
    )

    # 4. Overlay everything together (use cropped reference image if available)
    _ref_for_drawing = (
        REF_image_transformed_xr
        if REF_image_transformed_xr is not None
        else REF_image_xr
    )

    _plot = (
        _ref_for_drawing.hvplot.rgb(
            x="X",
            y="Y",
            bands="S",
            data_aspect=1,
            flip_yaxis=True,
            rasterize=True,
            title="Reference Image (Live Lasso Contour)",
            width=650,
            height=550,
            hover=False,
        )
        if (is_rgb and "S" in _ref_for_drawing.dims)
        else _ref_for_drawing.hvplot.image(
            x="X",
            y="Y",
            data_aspect=1,
            flip_yaxis=True,
            rasterize=True,
            cmap="viridis",
            width=650,
            height=550,
            robust=True,
            colorbar=False,
            hover=False,
        )
    )

    _he_combined = (_plot * _he_polys).opts(active_tools=["pan", "wheel_zoom"])

    _he_combined
    return (REF_poly_stream,)


@app.cell
def _():
    return


@app.cell
def _(REF_image_transformed_xr, REF_pixel_um, REF_poly_stream, is_rgb):
    # Check if manual ROI annotations exist
    _xs_list = REF_poly_stream.data.get("xs", [])
    _ys_list = REF_poly_stream.data.get("ys", [])

    # Filter out invalid polygons (< 3 points)
    _valid_polygons = [
        (xs, ys) for xs, ys in zip(_xs_list, _ys_list) if len(xs) >= 3
    ]

    if len(_valid_polygons) == 0:
        # No manual annotations - use Otsu thresholding
        _ref_for_mask = REF_image_transformed_xr

        # Convert to grayscale if RGB
        if is_rgb and "S" in _ref_for_mask.dims:
            _gray = rgb_to_grayscale(_ref_for_mask, method="luminosity")
        else:
            _gray = _ref_for_mask

        mask_xr = mask_from_threshold(_gray, method="otsu")

        # No individual ROI centers in automatic mode
        roi_centers = []

        mo.output.append(
            mo.md(
                "**Automatic ROI detection** using Otsu thresholding (no manual annotations provided)"
            ).callout(kind="info")
        )
    else:
        # Manual annotations exist - use polygon drawing
        mask_xr = mask_from_polygons(
            _valid_polygons,
            REF_image_transformed_xr.coords["X"].values,
            REF_image_transformed_xr.coords["Y"].values,
        )

        # Get coordinate offsets (transformed image may not start at 0), for
        # the centroid marker overlay below
        _x_min = float(REF_image_transformed_xr.coords["X"].min())
        _y_min = float(REF_image_transformed_xr.coords["Y"].min())

        roi_centers = []
        for _xs, _ys in _valid_polygons:
            # Convert physical coordinates (μm) to pixel indices
            _pts_physical = np.column_stack((_xs, _ys))
            _pts_px = (
                _pts_physical - np.array([_x_min, _y_min])
            ) / REF_pixel_um
            roi_centers.append(
                (np.mean(_pts_px[:, 0]), np.mean(_pts_px[:, 1]))
            )

    # Visualization
    _mask_plot = mask_xr.hvplot.image(
        x="X",
        y="Y",
        cmap="viridis",
        title="ROI Annotation Mask (Ground Truth)",
        data_aspect=1,
        flip_yaxis=True,
    )

    # Convert pixel centers back to physical coordinates (only if manual annotations)
    if len(roi_centers) > 0:
        _x_min = float(REF_image_transformed_xr.coords["X"].min())
        _y_min = float(REF_image_transformed_xr.coords["Y"].min())
        _phys_centers = [
            (c[0] * REF_pixel_um + _x_min, c[1] * REF_pixel_um + _y_min)
            for c in roi_centers
        ]
        _centers_plot = hv.Points(_phys_centers).opts(color="red", size=5)
        mask_with_centers = _mask_plot * _centers_plot
    else:
        mask_with_centers = _mask_plot

    mask_with_centers
    return (mask_xr,)


@app.cell
def _(MALDI_image_transformed_xr, mask_xr):
    # Get the first m/z and first two iterations for comparison
    if MALDI_image_transformed_xr is not None:
        _first_mz = MALDI_image_transformed_xr.M_Z.values[0]
        _iterations = MALDI_image_transformed_xr.ITERATION.values

        if len(_iterations) >= 2:
            MALDI_transform_0 = MALDI_image_transformed_xr.sel(
                M_Z=_first_mz, ITERATION=_iterations[0]
            )
            MALDI_transform_1 = MALDI_image_transformed_xr.sel(
                M_Z=_first_mz, ITERATION=_iterations[1]
            )
        elif len(_iterations) == 1:
            MALDI_transform_0 = MALDI_image_transformed_xr.sel(
                M_Z=_first_mz, ITERATION=_iterations[0]
            )
            MALDI_transform_1 = MALDI_image_transformed_xr.sel(
                M_Z=_first_mz, ITERATION=_iterations[0]
            )
        else:
            MALDI_transform_0 = None
            MALDI_transform_1 = None
    else:
        MALDI_transform_0 = None
        MALDI_transform_1 = None

    # Flatten arrays
    _y_true = mask_xr.values.flatten().astype(np.uint8)
    _y_scores_0 = (
        MALDI_transform_0.values.flatten()
        if MALDI_transform_0 is not None
        else np.zeros_like(_y_true, dtype=np.float64)
    )
    _y_scores_6 = (
        MALDI_transform_1.values.flatten()
        if MALDI_transform_1 is not None
        else np.zeros_like(_y_true, dtype=np.float64)
    )

    # ROC AUC
    auc_0 = roc_auc_score(_y_true, _y_scores_0)
    auc_6 = roc_auc_score(_y_true, _y_scores_6)

    _fpr_0, _tpr_0, _ = roc_curve(_y_true, _y_scores_0)
    _fpr_6, _tpr_6, _ = roc_curve(_y_true, _y_scores_6)

    _roc_df_0 = pl.DataFrame(
        {"FPR": _fpr_0, "TPR": _tpr_0, "Iteration": f"0 (AUC={auc_0:.3f})"}
    )
    _roc_df_6 = pl.DataFrame(
        {"FPR": _fpr_6, "TPR": _tpr_6, "Iteration": f"6 (AUC={auc_6:.3f})"}
    )
    _roc_df = pl.concat([_roc_df_0, _roc_df_6])

    roc_plot = _roc_df.hvplot.line(
        x="FPR",
        y="TPR",
        by="Iteration",
        title="ROC Curve",
        width=500,
        height=400,
    ) * hv.Curve([[0, 0], [1, 1]]).opts(color="black", line_dash="dashed")

    # Precision-recall metrics
    _precision_0, _recall_0, _thresholds_0 = precision_recall_curve(
        _y_true, _y_scores_0
    )
    _precision_6, _recall_6, _thresholds_6 = precision_recall_curve(
        _y_true, _y_scores_6
    )

    pr_auc_0 = auc(_recall_0[::-1], _precision_0[::-1])
    pr_auc_6 = auc(_recall_6[::-1], _precision_6[::-1])
    average_precision_0 = average_precision_score(_y_true, _y_scores_0)
    average_precision_6 = average_precision_score(_y_true, _y_scores_6)

    best_threshold_0, best_pixel_f1_0 = best_threshold_from_pr(
        _precision_0, _recall_0, _thresholds_0
    )
    best_threshold_6, best_pixel_f1_6 = best_threshold_from_pr(
        _precision_6, _recall_6, _thresholds_6
    )

    _pr_df_0 = pl.DataFrame(
        {
            "Recall": _recall_0[:-1],
            "Precision": _precision_0[:-1],
            "Iteration": f"0 (PR-AUC={pr_auc_0:.3f})",
        }
    )
    _pr_df_6 = pl.DataFrame(
        {
            "Recall": _recall_6[:-1],
            "Precision": _precision_6[:-1],
            "Iteration": f"6 (PR-AUC={pr_auc_6:.3f})",
        }
    )
    _pr_df = pl.concat([_pr_df_0, _pr_df_6])

    pr_plot = _pr_df.hvplot.line(
        x="Recall",
        y="Precision",
        by="Iteration",
        title="Precision-Recall Curve",
        width=500,
        height=400,
    )

    # Euclidean Distance Transform (EDT)
    edt_map = distance_transform_edt(mask_xr.values == 0)

    # Use EDT as an intensity-weighted metric: average distance to nearest ROI
    _sum_0 = np.sum(_y_scores_0)
    _sum_6 = np.sum(_y_scores_6)

    edt_metric_0 = (
        np.sum(_y_scores_0 * edt_map.flatten()) / _sum_0 if _sum_0 > 0 else 0
    )
    edt_metric_6 = (
        np.sum(_y_scores_6 * edt_map.flatten()) / _sum_6 if _sum_6 > 0 else 0
    )
    return (
        MALDI_transform_0,
        MALDI_transform_1,
        auc_0,
        auc_6,
        average_precision_0,
        average_precision_6,
        best_pixel_f1_0,
        best_pixel_f1_6,
        best_threshold_0,
        best_threshold_6,
        pr_auc_0,
        pr_auc_6,
        pr_plot,
        roc_plot,
    )


@app.cell
def _(pr_plot, roc_plot):
    hv.extension("matplotlib")
    (roc_plot + pr_plot).cols(2)
    return


@app.cell
def _(
    MALDI_transform_0,
    MALDI_transform_1,
    REF_image_transformed_xr,
    REF_image_xr,
    is_rgb,
    mask_xr,
):
    # Compute correlations and other metrics (use cropped reference)
    _ref_for_analysis = (
        REF_image_transformed_xr
        if REF_image_transformed_xr is not None
        else REF_image_xr
    )
    _mask_flat = mask_xr.values.flatten()
    _ref_flat = (
        rgb_to_grayscale(_ref_for_analysis, method="mean").values.flatten()
        if is_rgb and "S" in _ref_for_analysis.dims
        else _ref_for_analysis.values.flatten()
    )

    _maldi_0_flat = (
        MALDI_transform_0.values.flatten()
        if MALDI_transform_0 is not None
        else np.zeros_like(_mask_flat)
    )
    _maldi_6_flat = (
        MALDI_transform_1.values.flatten()
        if MALDI_transform_1 is not None
        else np.zeros_like(_mask_flat)
    )

    # Correlation with mask
    corr_mask_0 = np.corrcoef(_mask_flat, _maldi_0_flat)[0, 1]
    corr_mask_6 = np.corrcoef(_mask_flat, _maldi_6_flat)[0, 1]

    # Inside/outside ROIs
    _in_mask = _mask_flat > 0
    _out_mask = _mask_flat == 0

    _inside_0 = _maldi_0_flat[_in_mask]
    _inside_6 = _maldi_6_flat[_in_mask]
    _outside_0 = _maldi_0_flat[_out_mask]
    _outside_6 = _maldi_6_flat[_out_mask]

    if np.sum(_in_mask) > 1:
        corr_he_in_0 = np.corrcoef(_ref_flat[_in_mask], _inside_0)[0, 1]
        corr_he_in_6 = np.corrcoef(_ref_flat[_in_mask], _inside_6)[0, 1]

        cv_in_0 = (
            np.std(_inside_0) / np.mean(_inside_0)
            if np.mean(_inside_0) > 0
            else 0.0
        )
        cv_in_6 = (
            np.std(_inside_6) / np.mean(_inside_6)
            if np.mean(_inside_6) > 0
            else 0.0
        )
    else:
        corr_he_in_0, corr_he_in_6 = 0.0, 0.0
        cv_in_0, cv_in_6 = 0.0, 0.0

    if np.sum(_out_mask) > 1:
        corr_he_out_0 = np.corrcoef(_ref_flat[_out_mask], _outside_0)[0, 1]
        corr_he_out_6 = np.corrcoef(_ref_flat[_out_mask], _outside_6)[0, 1]
    else:
        corr_he_out_0, corr_he_out_6 = 0.0, 0.0

    # Correlation between iterations
    if np.sum(_in_mask) > 1:
        corr_0_vs_6_in = np.corrcoef(_inside_0, _inside_6)[0, 1]
    else:
        corr_0_vs_6_in = 0.0

    if np.sum(_out_mask) > 1:
        corr_0_vs_6_out = np.corrcoef(_outside_0, _outside_6)[0, 1]
    else:
        corr_0_vs_6_out = 0.0

    enrichment_ratio_0 = inside_outside_ratio(_inside_0, _outside_0)
    enrichment_ratio_6 = inside_outside_ratio(_inside_6, _outside_6)
    cohens_d_0 = cohens_d(_inside_0, _outside_0)
    cohens_d_6 = cohens_d(_inside_6, _outside_6)

    _ref_binned = digitize_for_nmi(_ref_flat)
    nmi_ref_0 = normalized_mutual_info_score(
        _ref_binned, digitize_for_nmi(_maldi_0_flat)
    )
    nmi_ref_6 = normalized_mutual_info_score(
        _ref_binned, digitize_for_nmi(_maldi_6_flat)
    )
    return (
        cohens_d_0,
        cohens_d_6,
        corr_0_vs_6_in,
        corr_0_vs_6_out,
        corr_he_in_0,
        corr_he_in_6,
        corr_he_out_0,
        corr_he_out_6,
        corr_mask_0,
        corr_mask_6,
        cv_in_0,
        cv_in_6,
        enrichment_ratio_0,
        enrichment_ratio_6,
        nmi_ref_0,
        nmi_ref_6,
    )


@app.cell
def _(
    MALDI_transform_0,
    MALDI_transform_1,
    REF_pixel_um,
    best_threshold_0,
    best_threshold_6,
    mask_xr,
):
    _gt_mask = mask_xr.values.astype(bool)

    (
        boundary_precision_0,
        boundary_recall_0,
        boundary_f1_0,
        mean_surface_distance_um_0,
        hausdorff95_um_0,
    ) = compute_boundary_metrics(
        MALDI_transform_0, best_threshold_0, _gt_mask, REF_pixel_um
    )
    (
        boundary_precision_6,
        boundary_recall_6,
        boundary_f1_6,
        mean_surface_distance_um_6,
        hausdorff95_um_6,
    ) = compute_boundary_metrics(
        MALDI_transform_1, best_threshold_6, _gt_mask, REF_pixel_um
    )
    return (
        boundary_f1_0,
        boundary_f1_6,
        hausdorff95_um_0,
        hausdorff95_um_6,
        mean_surface_distance_um_0,
        mean_surface_distance_um_6,
    )


@app.cell
def _(MALDI_transform_0, MALDI_transform_1, REF_pixel_um, mask_xr):
    _profile = radial_intensity_profile(
        mask_xr.values,
        {
            "Iteration_0": MALDI_transform_0.values,
            "Iteration_6": MALDI_transform_1.values,
        },
        pixel_size=REF_pixel_um,
    )

    _profile_0_arr = _profile["Iteration_0_mean"].to_numpy()
    _profile_6_arr = _profile["Iteration_6_mean"].to_numpy()

    _max_0 = (
        np.nanmax(_profile_0_arr)
        if np.any(np.isfinite(_profile_0_arr))
        else 1.0
    )
    _max_6 = (
        np.nanmax(_profile_6_arr)
        if np.any(np.isfinite(_profile_6_arr))
        else 1.0
    )
    _max_0 = _max_0 if _max_0 > 0 else 1.0
    _max_6 = _max_6 if _max_6 > 0 else 1.0

    radial_profile_df = _profile.rename(
        {
            "distance": "Distance_um",
            "Iteration_0_mean": "Iteration_0",
            "Iteration_6_mean": "Iteration_6",
        }
    ).with_columns(
        (pl.col("Iteration_0") / _max_0).alias("Iteration_0_norm"),
        (pl.col("Iteration_6") / _max_6).alias("Iteration_6_norm"),
        (pl.col("Iteration_0_std") / _max_0).alias("Iteration_0_norm_std"),
        (pl.col("Iteration_6_std") / _max_6).alias("Iteration_6_norm_std"),
    )
    return (radial_profile_df,)


@app.cell
def _(radial_profile_df):
    hv.extension("bokeh")

    _dist = radial_profile_df["Distance_um"].to_numpy()

    # Iteration 0 (Raw)
    _y0 = radial_profile_df["Iteration_0"].to_numpy()
    _std0 = radial_profile_df["Iteration_0_std"].to_numpy()
    _curve_0 = hv.Curve((_dist, _y0), label="Iteration 0").opts(
        color="blue", line_width=2
    )
    _spread_0 = hv.Spread((_dist, _y0, _std0)).opts(color="blue", alpha=0.15)

    # Iteration 6 (Deconvolved)
    _y6 = radial_profile_df["Iteration_6"].to_numpy()
    _std6 = radial_profile_df["Iteration_6_std"].to_numpy()
    _curve_6 = hv.Curve((_dist, _y6), label="Iteration 6").opts(
        color="orange", line_width=2
    )
    _spread_6 = hv.Spread((_dist, _y6, _std6)).opts(color="orange", alpha=0.15)

    radial_profile_plot = (_curve_0 * _spread_0 * _curve_6 * _spread_6).opts(
        title="Intensity Profile relative to Border (Raw ± SD)",
        xlabel="Distance from ROI Border (µm)",
        ylabel="Mean MALDI Intensity",
        width=450,
        height=350,
        legend_position="top_right",
    )

    radial_profile_plot
    return (radial_profile_plot,)


@app.cell
def _(
    MALDI_transform_0,
    MALDI_transform_1,
    REF_image_transformed_xr,
    REF_pixel_um,
    REF_poly_stream,
    mask_xr,
):
    _xs_list = REF_poly_stream.data.get("xs", [])
    _ys_list = REF_poly_stream.data.get("ys", [])
    _valid_polygons = [
        (xs, ys) for xs, ys in zip(_xs_list, _ys_list) if len(xs) >= 3
    ]
    _y_len, _x_len = mask_xr.shape

    # Get coordinate offsets for proper physical to pixel conversion
    _x_min = float(REF_image_transformed_xr.coords["X"].min())
    _y_min = float(REF_image_transformed_xr.coords["Y"].min())

    widths_0 = []
    widths_6 = []
    cvs_0 = []
    cvs_6 = []

    def _append_roi_metrics(_crop_mask, _patch_0, _patch_6):
        _inside_0 = _patch_0[_crop_mask == 1]
        _inside_6 = _patch_6[_crop_mask == 1]

        if len(_inside_0) > 1 and np.mean(_inside_0) > 0:
            cvs_0.append(np.std(_inside_0) / np.mean(_inside_0))
        if len(_inside_6) > 1 and np.mean(_inside_6) > 0:
            cvs_6.append(np.std(_inside_6) / np.mean(_inside_6))

        _profile = radial_intensity_profile(
            _crop_mask, {"patch_0": _patch_0, "patch_6": _patch_6}
        )
        _dist_centers_px = _profile["distance"].to_numpy()
        _p_0 = _profile["patch_0_mean"].to_numpy()
        _p_6 = _profile["patch_6_mean"].to_numpy()

        _high_0, _low_0 = edge_rise_width(
            _dist_centers_px, _p_0, normalize="self"
        )
        _high_6, _low_6 = edge_rise_width(
            _dist_centers_px, _p_6, normalize="self"
        )
        _w0 = (
            _low_0 - _high_0
            if _high_0 is not None and _low_0 is not None
            else None
        )
        _w6 = (
            _low_6 - _high_6
            if _high_6 is not None and _low_6 is not None
            else None
        )
        if _w0 is not None and _w0 >= 0:
            widths_0.append(_w0 * REF_pixel_um)
        if _w6 is not None and _w6 >= 0:
            widths_6.append(_w6 * REF_pixel_um)

    if len(_valid_polygons) > 0:
        for _xs, _ys in _valid_polygons:
            # Convert physical coordinates (μm) to pixel indices
            # Account for coordinate offset from cropping
            _pts_px = (
                np.column_stack((_xs, _ys)) - np.array([_x_min, _y_min])
            ) / REF_pixel_um

            _x0, _x1 = (
                int(np.floor(np.min(_pts_px[:, 0]))),
                int(np.ceil(np.max(_pts_px[:, 0]))),
            )
            _y0, _y1 = (
                int(np.floor(np.min(_pts_px[:, 1]))),
                int(np.ceil(np.max(_pts_px[:, 1]))),
            )

            # Pad by 30 pixels to capture background
            _pad = 30
            _x0_pad = max(0, _x0 - _pad)
            _x1_pad = min(_x_len, _x1 + _pad)
            _y0_pad = max(0, _y0 - _pad)
            _y1_pad = min(_y_len, _y1 + _pad)

            _crop_w = _x1_pad - _x0_pad
            _crop_h = _y1_pad - _y0_pad
            if _crop_w <= 0 or _crop_h <= 0:
                continue

            _patch_0_xr = MALDI_transform_0.isel(
                Y=slice(_y0_pad, _y1_pad), X=slice(_x0_pad, _x1_pad)
            )
            _patch_6_xr = MALDI_transform_1.isel(
                Y=slice(_y0_pad, _y1_pad), X=slice(_x0_pad, _x1_pad)
            )

            _crop_mask = mask_from_polygons(
                [(_xs, _ys)],
                _patch_0_xr.coords["X"].values,
                _patch_0_xr.coords["Y"].values,
            ).values

            _append_roi_metrics(
                _crop_mask, _patch_0_xr.values, _patch_6_xr.values
            )
    else:
        _labeled_mask, _n_components = label(mask_xr.values > 0)
        _component_slices = find_objects(_labeled_mask)
        _min_component_area_px = 25

        for _label_idx, _component_slice in enumerate(
            _component_slices, start=1
        ):
            if _component_slice is None:
                continue

            _crop_mask = (
                _labeled_mask[_component_slice] == _label_idx
            ).astype(np.uint8)
            if np.sum(_crop_mask) < _min_component_area_px:
                continue

            _patch_0 = MALDI_transform_0.isel(
                Y=_component_slice[0], X=_component_slice[1]
            ).values
            _patch_6 = MALDI_transform_1.isel(
                Y=_component_slice[0], X=_component_slice[1]
            ).values
            _append_roi_metrics(_crop_mask, _patch_0, _patch_6)

    edge_width_mean_0 = np.mean(widths_0) if len(widths_0) > 0 else 0.0
    edge_width_sd_0 = np.std(widths_0) if len(widths_0) > 0 else 0.0
    edge_width_mean_6 = np.mean(widths_6) if len(widths_6) > 0 else 0.0
    edge_width_sd_6 = np.std(widths_6) if len(widths_6) > 0 else 0.0

    cv_mean_0 = np.mean(cvs_0) if len(cvs_0) > 0 else 0.0
    cv_sd_0 = np.std(cvs_0) if len(cvs_0) > 0 else 0.0
    cv_mean_6 = np.mean(cvs_6) if len(cvs_6) > 0 else 0.0
    cv_sd_6 = np.std(cvs_6) if len(cvs_6) > 0 else 0.0
    return (
        cv_mean_0,
        cv_mean_6,
        cv_sd_0,
        cv_sd_6,
        cvs_0,
        cvs_6,
        edge_width_mean_0,
        edge_width_mean_6,
        edge_width_sd_0,
        edge_width_sd_6,
        widths_0,
        widths_6,
    )


@app.cell
def _(REF_pixel_um, radial_profile_df, widths_0, widths_6):
    hv.extension("bokeh")

    _dist_um = radial_profile_df["Distance_um"].to_numpy()
    _norm_0 = radial_profile_df["Iteration_0_norm"].to_numpy()
    _norm_6 = radial_profile_df["Iteration_6_norm"].to_numpy()

    _t90_0, _t10_0 = edge_rise_width(_dist_um, _norm_0, normalize="none")
    _t90_0 = _dist_um[-1] if _t90_0 is None else _t90_0
    _t10_0 = _dist_um[-1] if _t10_0 is None else _t10_0

    _t90_6, _t10_6 = edge_rise_width(_dist_um, _norm_6, normalize="none")
    _t90_6 = _dist_um[-1] if _t90_6 is None else _t90_6
    _t10_6 = _dist_um[-1] if _t10_6 is None else _t10_6

    lsf_fwhm_um_0 = lsf_fwhm(_dist_um, _norm_0)
    lsf_fwhm_um_6 = lsf_fwhm(_dist_um, _norm_6)
    lsf_fwhm_px_0 = (
        lsf_fwhm_um_0 / REF_pixel_um if np.isfinite(lsf_fwhm_um_0) else np.nan
    )
    lsf_fwhm_px_6 = (
        lsf_fwhm_um_6 / REF_pixel_um if np.isfinite(lsf_fwhm_um_6) else np.nan
    )

    # 1. Knife-edge test plot on global profile
    _curve_0 = hv.Curve((_dist_um, _norm_0), label="Iteration 0 (Raw)").opts(
        color="blue", line_width=2
    )
    _curve_6 = hv.Curve(
        (_dist_um, _norm_6), label="Iteration 6 (Deconvolved)"
    ).opts(color="orange", line_width=2)

    _hline_90 = hv.HLine(0.90).opts(
        color="gray", line_dash="dashed", line_width=1
    )
    _hline_10 = hv.HLine(0.10).opts(
        color="gray", line_dash="dashed", line_width=1
    )

    _vline_90_0 = hv.VLine(_t90_0).opts(
        color="blue", line_dash="dotted", line_width=1.5
    )
    _vline_10_0 = hv.VLine(_t10_0).opts(
        color="blue", line_dash="dotted", line_width=1.5
    )

    _vline_90_6 = hv.VLine(_t90_6).opts(
        color="orange", line_dash="dotted", line_width=1.5
    )
    _vline_10_6 = hv.VLine(_t10_6).opts(
        color="orange", line_dash="dotted", line_width=1.5
    )

    _edge_plot = (
        _curve_0
        * _curve_6
        * _hline_90
        * _hline_10
        * _vline_90_0
        * _vline_10_0
        * _vline_90_6
        * _vline_10_6
    ).opts(
        title="Knife-Edge Test (90% to 10% Intensity Drop)",
        xlabel="Distance from ROI Border (µm)",
        ylabel="Normalized Intensity",
        width=450,
        height=350,
        legend_position="top_right",
    )

    # 2. Boxplot comparing the distribution of individual ROI edge widths
    _df_widths_0 = pl.DataFrame(
        {"Edge Rise Distance (µm)": widths_0, "Iteration": "Iteration 0"}
    )
    _df_widths_6 = pl.DataFrame(
        {"Edge Rise Distance (µm)": widths_6, "Iteration": "Iteration 6"}
    )
    _df_widths = pl.concat([_df_widths_0, _df_widths_6])

    _edge_boxplot = _df_widths.hvplot.box(
        y="Edge Rise Distance (µm)",
        by="Iteration",
        title="Individual ROI Edge Widths",
        legend=False,
        width=450,
        height=350,
        color="Iteration",
        cmap=["blue", "orange"],
    )

    edge_rise_plots = _edge_plot + _edge_boxplot

    edge_width_um_0 = _t10_0 - _t90_0
    edge_width_px_0 = int(np.round(edge_width_um_0 / REF_pixel_um))
    edge_width_um_6 = _t10_6 - _t90_6
    edge_width_px_6 = int(np.round(edge_width_um_6 / REF_pixel_um))
    return (
        edge_rise_plots,
        edge_width_px_0,
        edge_width_px_6,
        edge_width_um_0,
        edge_width_um_6,
        lsf_fwhm_px_0,
        lsf_fwhm_px_6,
        lsf_fwhm_um_0,
        lsf_fwhm_um_6,
    )


@app.cell
def _(
    REF_pixel_um,
    auc_0,
    auc_6,
    average_precision_0,
    average_precision_6,
    best_pixel_f1_0,
    best_pixel_f1_6,
    boundary_f1_0,
    boundary_f1_6,
    cohens_d_0,
    cohens_d_6,
    corr_0_vs_6_in,
    corr_0_vs_6_out,
    corr_he_in_0,
    corr_he_in_6,
    corr_he_out_0,
    corr_he_out_6,
    corr_mask_0,
    corr_mask_6,
    cv_in_0,
    cv_in_6,
    cv_mean_0,
    cv_mean_6,
    cv_sd_0,
    cv_sd_6,
    cvs_0,
    cvs_6,
    edge_rise_plots,
    edge_width_mean_0,
    edge_width_mean_6,
    edge_width_px_0,
    edge_width_px_6,
    edge_width_sd_0,
    edge_width_sd_6,
    edge_width_um_0,
    edge_width_um_6,
    enrichment_ratio_0,
    enrichment_ratio_6,
    hausdorff95_um_0,
    hausdorff95_um_6,
    lsf_fwhm_px_0,
    lsf_fwhm_px_6,
    lsf_fwhm_um_0,
    lsf_fwhm_um_6,
    mean_surface_distance_um_0,
    mean_surface_distance_um_6,
    nmi_ref_0,
    nmi_ref_6,
    pr_auc_0,
    pr_auc_6,
    radial_profile_plot,
    widths_0,
    widths_6,
):
    _display_cv_mean_0 = cv_mean_0 if len(cvs_0) > 0 else cv_in_0
    _display_cv_sd_0 = cv_sd_0 if len(cvs_0) > 1 else 0.0
    _display_cv_mean_6 = cv_mean_6 if len(cvs_6) > 0 else cv_in_6
    _display_cv_sd_6 = cv_sd_6 if len(cvs_6) > 1 else 0.0

    _display_edge_width_mean_0 = (
        edge_width_mean_0 if len(widths_0) > 0 else edge_width_um_0
    )
    _display_edge_width_sd_0 = edge_width_sd_0 if len(widths_0) > 1 else 0.0
    _display_edge_width_mean_6 = (
        edge_width_mean_6 if len(widths_6) > 0 else edge_width_um_6
    )
    _display_edge_width_sd_6 = edge_width_sd_6 if len(widths_6) > 1 else 0.0

    _report = mo.md(
        f"""
        ### Evaluation Metrics (ROI Annotations vs. MALDI)

        | Metric | Iteration 0 (Raw) | Iteration 6 (Deconvolved) | Description |
        |---|---|---|---|
        | **ROC-AUC** | `{auc_0:.4f}` | `{auc_6:.4f}` | Area under the ROC curve (higher is better) |
        | **PR-AUC** | `{pr_auc_0:.4f}` | `{pr_auc_6:.4f}` | Precision-recall area under curve (better for sparse ROIs) |
        | **Average Precision** | `{average_precision_0:.4f}` | `{average_precision_6:.4f}` | Average precision summary of detection quality |
        | **Best Pixel F1** | `{best_pixel_f1_0:.4f}` | `{best_pixel_f1_6:.4f}` | Best F1 over MALDI thresholds against ROI mask |
        | **Intra-ROI CV** | `{_display_cv_mean_0:.4f} ± {_display_cv_sd_0:.4f}` | `{_display_cv_mean_6:.4f} ± {_display_cv_sd_6:.4f}` | Coefficient of variation inside ROIs (lower is smoother) |
        | **Inside/Outside Enrichment** | `{enrichment_ratio_0:.4f}` | `{enrichment_ratio_6:.4f}` | Mean ROI intensity divided by mean background intensity |
        | **Cohen's d (In vs. Out)** | `{cohens_d_0:.4f}` | `{cohens_d_6:.4f}` | Effect size separating ROI and background intensities |
        | **Boundary F1 (@ 2 px)** | `{boundary_f1_0:.4f}` | `{boundary_f1_6:.4f}` | ROI-boundary agreement after thresholding MALDI at best pixel F1 |
        | **Mean Surface Distance** | `{mean_surface_distance_um_0:.2f} µm` | `{mean_surface_distance_um_6:.2f} µm` | Mean bidirectional boundary distance (lower is better) |
        | **95% Hausdorff Distance** | `{hausdorff95_um_0:.2f} µm` | `{hausdorff95_um_6:.2f} µm` | Robust worst-case boundary mismatch (lower is better) |
        | **Edge Rise Distance** | `{_display_edge_width_mean_0:.2f} ± {_display_edge_width_sd_0:.2f} µm` (`{_display_edge_width_mean_0 / REF_pixel_um:.1f} ± {_display_edge_width_sd_0 / REF_pixel_um:.1f} px`) | `{_display_edge_width_mean_6:.2f} ± {_display_edge_width_sd_6:.2f} µm` (`{_display_edge_width_mean_6 / REF_pixel_um:.1f} ± {_display_edge_width_sd_6 / REF_pixel_um:.1f} px`) | Mean individual ROI edge width (90% to 10% drop, lower is sharper) |
        | *Global Edge Rise* | `{edge_width_um_0:.2f} µm` ({edge_width_px_0} px) | `{edge_width_um_6:.2f} µm` ({edge_width_px_6} px) | Global transition width (90% to 10% drop on average profile) |
        | **LSF FWHM** | `{lsf_fwhm_um_0:.2f} µm` (`{lsf_fwhm_px_0:.1f} px`) | `{lsf_fwhm_um_6:.2f} µm` (`{lsf_fwhm_px_6:.1f} px`) | Full width at half maximum of the line-spread function |
        | **Normalized Mutual Information** | `{nmi_ref_0:.4f}` | `{nmi_ref_6:.4f}` | Binned NMI between reference image and MALDI intensity map |

        #### Pearson Correlation Coefficients

        | Metric | Iteration 0 | Iteration 6 | Description |
        |---|---|---|---|
        | **Correlation with Mask (In vs. Out)** | `{corr_mask_0:.4f}` | `{corr_mask_6:.4f}` | Correlation between MALDI signal and binary ROI mask |
        | **Correlation with Reference (Inside ROIs)** | `{corr_he_in_0:.4f}` | `{corr_he_in_6:.4f}` | Correlation between MALDI and reference image inside ROIs |
        | **Correlation with Reference (Outside ROIs)** | `{corr_he_out_0:.4f}` | `{corr_he_out_6:.4f}` | Correlation between MALDI and reference image outside ROIs |

        * **Pearson Correlation between Iteration 0 and Iteration 6:**
          * Inside ROIs: `{corr_0_vs_6_in:.4f}`
          * Outside ROIs: `{corr_0_vs_6_out:.4f}`
        """
    ).callout()

    _plots = mo.vstack(
        [
            mo.md("#### Radial Intensity Profile relative to ROI Border"),
            radial_profile_plot,
            mo.md("#### Edge Rise Distance (Knife-Edge Test & Boxplot)"),
            edge_rise_plots,
            # mo.md("#### Sobel Gradients (Visualizing Edge Sharpness/Smoothness)"),
            # sobel_edges_plot,
        ]
    )
    mo.vstack([_report, _plots])
    return


if __name__ == "__main__":
    app.run()
