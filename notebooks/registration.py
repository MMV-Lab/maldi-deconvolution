import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    import polars as pl
    import numpy as np
    import xarray as xr

    import tifffile
    from pathlib import Path
    from bioio import BioImage
    import bioio_bioformats

    import matplotlib as mpl
    import panel as pn
    import holoviews as hv
    import hvplot
    import hvplot.xarray
    import hvplot.polars
    from datashader.transfer_functions import shade, stack
    from matplotlib.colors import LinearSegmentedColormap

    hv.extension("bokeh")

    from PIL import Image, ImageChops, ImageDraw

    from pyimzml.ImzMLParser import ImzMLParser
    import SimpleITK as sitk


@app.cell
def _():
    ref_select_widget = mo.ui.file_browser(
        multiple=False,
        selection_mode="file",
        filetypes=[".tif"],
    )
    maldi_select_widget = mo.ui.file_browser(
        multiple=False,
        selection_mode="file",
        filetypes=[".h5"],
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
                        "# Select the **h5** file containing the **deconvolved MALDI** data:"
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

    if REF_image.physical_pixel_sizes.X == REF_image.physical_pixel_sizes.Y:
        _tmp_pixel_size = str(REF_image.physical_pixel_sizes.X)

    REF_pixel_size_text = mo.ui.text(value=_tmp_pixel_size)

    deconv_image_path = maldi_select_widget.value[0].path
    deconv_image = xr.load_dataarray(deconv_image_path)

    _mz = deconv_image.m_z.values.tolist()
    _iter = deconv_image.iteration.values.tolist()

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
            label="Fluorescence Channel:"
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
        mo.md(
            f"Pixel Size Reference Image: {REF_pixel_size_text} μm / pixel"
        ),
    ]
    if is_fluorescence and REF_channel_widget is not None:
        _ref_ui_elements.append(REF_channel_widget)
        _ref_ui_elements.append(
            mo.md(f"**Detected: Fluorescence image** ({len(_channels)} channels)")
        )
    elif is_rgb:
        _ref_ui_elements.append(
            mo.md("**Detected: RGB/H&E image**")
        )

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
        iteration=MALDI_iteration_registration,
        m_z=MALDI_m_z_registration,
    ).rename({"y": "Y", "x": "X"})

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
    hvplot.extension("bokeh")

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
        _REF_base = REF_image_xr.hvplot.image(
            x="X",
            y="Y",
            data_aspect=1,
            flip_yaxis=True,
            rasterize=True,
            cmap="viridis",
            title="Reference Image (Fluorescence)",
            width=650,
            height=550,
            hover=False,
            robust=True,
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
        shared_axes=False
    )
    landmark_selection_plot
    return MALDI_lm_stream, REF_lm_stream


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
def _(MALDI_lm_stream, REF_lm_stream):
    # fucci fig 3

    REF_lm_stream.event(data=
      {
        "x": [
          107.7567620025882,
          100.76429197900602,
          126.23682596658055,
          102.26265391598334,
          198.65875799146863,
          147.71369001631956,
          138.72344218266363,
          223.13238402105827
        ],
        "y": [
          1033.8688376856246,
          996.1450969138434,
          1008.8897721201462,
          1068.534603166206,
          985.4390719016737,
          939.0485786104502,
          913.6867994096265,
          1002.2622920934311
        ]
      }
    )

    MALDI_lm_stream.event(data={
        "x": [
          64.72998639809967,
          59.69391720241973,
          83.95883016412911,
          59.69391720241973,
          154.4643577760639,
          108.22367326678655,
          101.81410525081137,
          182.84969733960705
        ],
        "y": [
          166.04315909067785,
          126.73811618804928,
          141.37419883728023,
          193.78111481984453,
          119.65608493804928,
          82.35727175394689,
          46.9471155039469,
          140.42958220163976
        ]
      }
    )
    return


@app.cell
def _():
    # (
    #     REF_lm_stream.data,
    #     MALDI_lm_stream.data
    # )
    return


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
        _REF_gray = (
            0.299 * REF_image_xr.sel(S="R").values.astype(np.float32)
            + 0.587 * REF_image_xr.sel(S="G").values.astype(np.float32)
            + 0.114 * REF_image_xr.sel(S="B").values.astype(np.float32)
        )
    else:
        # Fluorescence or grayscale: use directly
        _REF_gray = REF_image_xr.values.astype(np.float32)

    REF_sitk = sitk.GetImageFromArray(_REF_gray)
    REF_sitk.SetSpacing((REF_pixel_um, REF_pixel_um))

    # MALDI 2D array
    MALDI_sitk = sitk.GetImageFromArray(
        MALDI_image_registration_xr.values.astype(np.float32)
    )
    MALDI_sitk.SetSpacing(
        (
            MALDI_image_registration_xr.attrs["pixel_size_x"].item(),
            MALDI_image_registration_xr.attrs["pixel_size_y"].item(),
        )
    )

    # Landmark-based affine initialisation (coordinates are already in µm)
    _fixed_lm = [
        v for pt in zip(_REF_x, _REF_y) for v in (float(pt[0]), float(pt[1]))
    ]
    _moving_lm = [
        v
        for pt in zip(_MALDI_x, _MALDI_y)
        for v in (float(pt[0]), float(pt[1]))
    ]

    # Set up the LandmarkBasedTransformInitializerFilter.
    landmark_initializer = sitk.LandmarkBasedTransformInitializerFilter()

    landmark_initializer.SetFixedLandmarks(_fixed_lm)
    landmark_initializer.SetMovingLandmarks(_moving_lm)

    transform = sitk.AffineTransform(2)

    mo.md(
        f"Landmark affine transform initialised from **{len(_REF_x)} point pairs**."
    ).callout(kind="success")
    return MALDI_sitk, REF_sitk, landmark_initializer, transform


@app.cell
def _(MALDI_sitk, REF_sitk, is_rgb, landmark_initializer, transform):
    # if (deconv_image_path.parent / "transorm.tfm").is_file():
    #     REF_MALDI_transform = sitk.ReadTransform("transorm.tfm")
    #     mo.output.append(
    #         mo.md(f"""
    #     **Read from file**
    #     """).callout(kind="success")
    #     )
    # else:
    _reg = sitk.ImageRegistrationMethod()

    # Mattes Mutual Information: standard for multi-modal registration
    if is_rgb:
        _reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    else:
        _reg.SetMetricAsCorrelation()
    _reg.SetInterpolator(sitk.sitkLinear)

    # Gradient descent with line search
    _reg.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-7,
        convergenceWindowSize=10,
    )
    _reg.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution pyramid (coarse → fine)
    _reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    _reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0, 0.0])
    _reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    _output_landmark = landmark_initializer.Execute(transform)

    # Start from the landmark-based affine transform
    _reg.SetInitialTransform(_output_landmark, inPlace=False)

    _REF_norm = sitk.RescaleIntensity(sitk.Cast(REF_sitk, sitk.sitkFloat32))
    _MALDI_norm = sitk.RescaleIntensity(
        sitk.Cast(MALDI_sitk, sitk.sitkFloat32)
    )

    REF_MALDI_transform = _reg.Execute(_REF_norm, _MALDI_norm)

    mo.output.append(
        mo.md(f"""
    **Intensity-based registration complete**  
    Final metric value: `{_reg.GetMetricValue():.5f}`  
    Stopping condition: `{_reg.GetOptimizerStopConditionDescription()}`
    """).callout(kind="success")
    )
    return (REF_MALDI_transform,)


@app.cell
def _(REF_MALDI_transform, deconv_image_path):
    # Save the transform to file
    sitk.WriteTransform(
        REF_MALDI_transform, deconv_image_path.parent / "transorm.tfm"
    )
    return


@app.cell
def _(
    MALDI_iterations_transform,
    MALDI_m_z_transform,
    REF_MALDI_transform,
    REF_image_xr,
    REF_pixel_um,
    REF_sitk,
    deconv_image_path,
):
    # Use the transform widget values instead of hardcoded values
    _deconv_image = xr.load_dataarray(deconv_image_path)

    def register_MALDI(MALDI_raw):
        MALDI_sitk = sitk.GetImageFromArray(
            MALDI_raw.values.astype(np.float32)
        )
        MALDI_sitk.SetSpacing(
            (
                MALDI_raw.attrs["pixel_size_x"].item(),
                MALDI_raw.attrs["pixel_size_y"].item(),
            )
        )

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(REF_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        resampler.SetTransform(REF_MALDI_transform)
        MALDI_reg_sitk = resampler.Execute(MALDI_sitk)

        MALDI_arr = sitk.GetArrayFromImage(MALDI_reg_sitk).astype(np.float32)
        return MALDI_arr

    _n_mz = len(MALDI_m_z_transform)
    _n_iter = len(MALDI_iterations_transform)

    if _n_mz > 0 and _n_iter > 0:
        # Get Y, X dimensions from first registration
        _first_maldi = _deconv_image.sel(
            iteration=MALDI_iterations_transform[0], m_z=MALDI_m_z_transform[0]
        ).rename({"y": "Y", "x": "X"})
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
                    _MALDI_raw = _deconv_image.sel(
                        iteration=_iter, m_z=_mz
                    ).rename({"y": "Y", "x": "X"})
                    _registered_arr = register_MALDI(_MALDI_raw)
                    _all_arrays[i_mz, i_iter, :, :] = _registered_arr

        MALDI_image_transformed_xr = xr.DataArray(
            _all_arrays,
            dims=["m_z", "iteration", "Y", "X"],
            coords={
                "m_z": MALDI_m_z_transform,
                "iteration": MALDI_iterations_transform,
                "Y": REF_image_xr.coords["Y"],
                "X": REF_image_xr.coords["X"],
            },
            attrs={
                "pixel_size_x": REF_pixel_um,
                "pixel_size_y": REF_pixel_um,
                "long_name": "Ion Intensity",
                "units": "a.u.",
            },
        )
    else:
        MALDI_image_transformed_xr = None


    # --- Crop both REF and MALDI images to the transformed MALDI bounding box ---

    if MALDI_image_transformed_xr is not None:
        # 1. Get the physical bounds of the original MALDI image in its own coordinate space
        _size_x = int(_deconv_image.sizes["x"])
        _size_y = int(_deconv_image.sizes["y"])
        _spacing_x = float(_deconv_image.attrs["pixel_size_x"])
        _spacing_y = float(_deconv_image.attrs["pixel_size_y"])

        maldi_corners_local = [
            (0.0, 0.0),
            (_size_x * _spacing_x, 0.0),
            (0.0, _size_y * _spacing_y),
            (_size_x * _spacing_x, _size_y * _spacing_y)
        ]

        # 2. Map the local corners to the reference frame using the inverse transform
        try:
            inverse_transform = REF_MALDI_transform.GetInverse()
        except Exception as e:
            raise RuntimeError("Could not invert the transform to map MALDI coordinates to REF space.") from e

        ref_corners_physical = [inverse_transform.TransformPoint(pt) for pt in maldi_corners_local]

        # Extract mapped X and Y coordinates
        ref_xs = [pt[0] for pt in ref_corners_physical]
        ref_ys = [pt[1] for pt in ref_corners_physical]

        min_physical_x, max_physical_x = min(ref_xs), max(ref_xs)
        min_physical_y, max_physical_y = min(ref_ys), max(ref_ys)

        # 3. Handle coordinate directions (ascending vs descending coordinate axes)
        def get_slice_bounds(coords, min_val, max_val):
            if coords[0] < coords[-1]:  # Ascending
                return slice(min_val, max_val)
            else:  # Descending
                return slice(max_val, min_val)

        ref_x_coords = REF_image_xr.coords["X"].values
        ref_y_coords = REF_image_xr.coords["Y"].values

        x_slice = get_slice_bounds(ref_x_coords, min_physical_x, max_physical_x)
        y_slice = get_slice_bounds(ref_y_coords, min_physical_y, max_physical_y)

        # 4. Crop using standard xarray slicing (WITHOUT method="nearest")
        REF_image_transformed_xr = REF_image_xr.sel(
            Y=y_slice, 
            X=x_slice
        )

        MALDI_image_transformed_xr = MALDI_image_transformed_xr.sel(
            Y=y_slice, 
            X=x_slice
        )


    return MALDI_image_transformed_xr, REF_image_transformed_xr


@app.cell
def _(MALDI_image_transformed_xr, MALDI_m_z_registration, deconv_image_path):
    if MALDI_image_transformed_xr is not None:
        MALDI_image_transformed_xr.to_netcdf(
            deconv_image_path.parent
            / f"{deconv_image_path.name}_deconvolved_registered_m_z-{MALDI_m_z_registration}.h5"
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
    hvplot.extension("bokeh")

    linear_configs = {
        "Linear Red": ["black", "red"],
        "Linear Green": ["black", "#00ff00"],
        "Linear Blue": ["black", "blue"],
        "Linear Cyan": ["black", "cyan"],
        "Linear Magenta": ["black", "magenta"],
        "Linear Yellow": ["black", "yellow"],
    }

    cmap_dict = {}
    for name in linear_configs.keys():
        cmap_dict[name] = LinearSegmentedColormap.from_list(
            name, linear_configs[name]
        )


    if MALDI_m_z_registration in MALDI_image_transformed_xr.m_z.values:
        _MALDI_for_overlay = MALDI_image_transformed_xr.sel(m_z=MALDI_m_z_registration)
    else:
        # Fallback to first m/z if registration m/z not in transform selection
        _MALDI_for_overlay = MALDI_image_transformed_xr.isel(m_z=0)

    # Select last iteration to show final deconvolved result
    _MALDI_for_overlay = _MALDI_for_overlay.isel(iteration=-1)

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

        _REF_gray = REF_image_transformed_xr.mean(dim="S").hvplot.image(
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
        hv.opts.Overlay(title="Registration result: Cropped Reference + MALDI overlay (final iteration)")
    ) + (_REF_gray * _MALDI_gray).opts(
        hv.opts.Overlay(title="Registration result: Cropped Reference + MALDI overlay (final iteration)")
        )
    return


@app.cell
def _(REF_image_transformed_xr, REF_image_xr, is_rgb):
    hvplot.extension("bokeh")

    from scipy.interpolate import splprep, splev

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
            x_raw = np.array(xs)
            y_raw = np.array(ys)

            if len(x_raw) < 5:  # Need enough points to estimate derivatives
                paths.append(hv.Path((x_raw, y_raw)))
                continue

            # 1. FIX ARTIFACTS: Check if the loop is unclosed and gap-fill it
            dist_start_end = np.sqrt(
                (x_raw[0] - x_raw[-1]) ** 2 + (y_raw[0] - y_raw[-1]) ** 2
            )

            # If the gap is small (e.g., within 40 pixels), snap it shut
            if dist_start_end > 0 and dist_start_end < 40:
                x_raw = np.append(x_raw, x_raw[0])
                y_raw = np.append(y_raw, y_raw[0])

            # Remove duplicate adjacent points (standing still with the mouse)
            # to prevent singular matrices in parametric fitting
            keep = np.insert(
                np.sqrt(np.diff(x_raw) ** 2 + np.diff(y_raw) ** 2) > 1e-5,
                0,
                True,
            )
            x_raw, y_raw = x_raw[keep], y_raw[keep]

            if len(x_raw) < 5:
                continue

            # 2. PRO NUCLEAR SEGMENTATION: Use splprep with smoothing
            try:
                # per=True enforces a perfect closed-loop boundary condition
                # s controls smoothing. s=0 intercepts every point exactly.
                # Increasing 's' slightly (e.g., 5 to 20) filters out shaky hands.
                tck, u = splprep([x_raw, y_raw], s=7, per=True)

                # 3. Sample smoothly along the parametric space
                u_new = np.linspace(0, 1, 300)
                x_smooth, y_smooth = splev(u_new, tck)
                paths.append(hv.Path((x_smooth, y_smooth)))
            except Exception:
                # Fallback if the coordinates are too degenerate for splprep
                paths.append(hv.Path((x_raw, y_raw)))

        return hv.Path(paths).opts(line_color="cyan", line_width=2.5)

    # 3. Wrap the callback function into a DynamicMap bound to your stream
    _smooth_contour_dynamic = hv.DynamicMap(
        update_spline, streams=[REF_poly_stream]
    )

    # 4. Overlay everything together (use cropped reference image if available)
    _ref_for_drawing = REF_image_transformed_xr if REF_image_transformed_xr is not None else REF_image_xr

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
    from skimage.filters import threshold_otsu, threshold_li, threshold_triangle, threshold_isodata

    # Check if manual ROI annotations exist
    _xs_list = REF_poly_stream.data.get("xs", [])
    _ys_list = REF_poly_stream.data.get("ys", [])

    # Filter out invalid polygons (< 3 points)
    _valid_polygons = [(xs, ys) for xs, ys in zip(_xs_list, _ys_list) if len(xs) >= 3]

    if len(_valid_polygons) == 0:
        # No manual annotations - use Otsu thresholding
        _ref_for_mask = REF_image_transformed_xr
    
        # Convert to grayscale if RGB
        if is_rgb and "S" in _ref_for_mask.dims:
            _gray = (
                0.299 * _ref_for_mask.sel(S="R").values
                + 0.587 * _ref_for_mask.sel(S="G").values
                + 0.114 * _ref_for_mask.sel(S="B").values
            )
        else:
            _gray = _ref_for_mask.values
    
        # Apply Otsu thresholding
        _otsu_thresh = threshold_triangle(_gray)
        roi_mask = (_gray > _otsu_thresh).astype(np.uint8)
    
        # No individual ROI centers in automatic mode
        roi_centers = []
    
        mo.output.append(
            mo.md("**Automatic ROI detection** using Otsu thresholding (no manual annotations provided)").callout(kind="info")
        )
    else:
        # Manual annotations exist - use polygon drawing
        _y_len = REF_image_transformed_xr.sizes["Y"]
        _x_len = REF_image_transformed_xr.sizes["X"]
        _mask_img = Image.new("L", (_x_len, _y_len), 0)
        _draw = ImageDraw.Draw(_mask_img)

        roi_centers = []

        # Get coordinate offsets (transformed image may not start at 0)
        _x_min = float(REF_image_transformed_xr.coords["X"].min())
        _y_min = float(REF_image_transformed_xr.coords["Y"].min())

        for _xs, _ys in _valid_polygons:
            # Convert physical coordinates (μm) to pixel indices
            # Account for coordinate offset from cropping
            _pts_physical = np.column_stack((_xs, _ys))
            _pts_px = (_pts_physical - np.array([_x_min, _y_min])) / REF_pixel_um
            _px_x = _pts_px[:, 0]
            _px_y = _pts_px[:, 1]

            _polygon = [(x, y) for x, y in zip(_px_x, _px_y)]
            _draw.polygon(_polygon, outline=1, fill=1)

            # Store center in pixel coordinates
            _center_x = np.mean(_px_x)
            _center_y = np.mean(_px_y)
            roi_centers.append((_center_x, _center_y))

        roi_mask = np.array(_mask_img)

    # Create xarray mask (common to both paths)
    _ref_for_mask = REF_image_transformed_xr
    mask_xr = xr.DataArray(
        roi_mask,
        dims=["Y", "X"],
        coords={"X": _ref_for_mask.coords["X"], "Y": _ref_for_mask.coords["Y"]},
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
    from sklearn.metrics import roc_auc_score, roc_curve
    from scipy.ndimage import distance_transform_edt

    # Get the first m/z and first two iterations for comparison
    if MALDI_image_transformed_xr is not None:
        _first_mz = MALDI_image_transformed_xr.m_z.values[0]
        _iterations = MALDI_image_transformed_xr.iteration.values

        if len(_iterations) >= 2:
            MALDI_transform_0 = MALDI_image_transformed_xr.sel(
                m_z=_first_mz, iteration=_iterations[0]
            )
            MALDI_transform_1 = MALDI_image_transformed_xr.sel(
                m_z=_first_mz, iteration=_iterations[1]
            )
        elif len(_iterations) == 1:
            MALDI_transform_0 = MALDI_image_transformed_xr.sel(
                m_z=_first_mz, iteration=_iterations[0]
            )
            MALDI_transform_1 = MALDI_image_transformed_xr.sel(
                m_z=_first_mz, iteration=_iterations[0]
            )
        else:
            MALDI_transform_0 = None
            MALDI_transform_1 = None
    else:
        MALDI_transform_0 = None
        MALDI_transform_1 = None

    # Flatten arrays
    _y_true = mask_xr.values.flatten()
    _y_scores_0 = (
        MALDI_transform_0.values.flatten()
        if MALDI_transform_0 is not None
        else np.zeros_like(_y_true)
    )
    _y_scores_6 = (
        MALDI_transform_1.values.flatten()
        if MALDI_transform_1 is not None
        else np.zeros_like(_y_true)
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

    # Euclidean Distance Transform (EDT)
    # Computed on the ground truth mask (distance to nearest ROI boundary/center)
    # EDT is 0 inside ROIs, >0 outside.
    edt_map = distance_transform_edt(mask_xr.values == 0)

    # Use EDT as an intensity-weighted metric: average distance to nearest ROI
    # Lower is better (signal is concentrated in or near ROIs)
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
        distance_transform_edt,
        roc_plot,
    )


@app.cell
def _(roc_plot):
    hv.extension("matplotlib")
    roc_plot
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
    _ref_for_analysis = REF_image_transformed_xr if REF_image_transformed_xr is not None else REF_image_xr
    _mask_flat = mask_xr.values.flatten()
    _ref_flat = (
        _ref_for_analysis.mean(dim="S").values.flatten()
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

    if np.sum(_in_mask) > 1:
        corr_he_in_0 = np.corrcoef(
            _ref_flat[_in_mask], _maldi_0_flat[_in_mask]
        )[0, 1]
        corr_he_in_6 = np.corrcoef(
            _ref_flat[_in_mask], _maldi_6_flat[_in_mask]
        )[0, 1]

        # CV inside ROIs
        cv_in_0 = (
            np.std(_maldi_0_flat[_in_mask]) / np.mean(_maldi_0_flat[_in_mask])
            if np.mean(_maldi_0_flat[_in_mask]) > 0
            else 0.0
        )
        cv_in_6 = (
            np.std(_maldi_6_flat[_in_mask]) / np.mean(_maldi_6_flat[_in_mask])
            if np.mean(_maldi_6_flat[_in_mask]) > 0
            else 0.0
        )
    else:
        corr_he_in_0, corr_he_in_6 = 0.0, 0.0
        cv_in_0, cv_in_6 = 0.0, 0.0

    _out_mask = _mask_flat == 0
    if np.sum(_out_mask) > 1:
        corr_he_out_0 = np.corrcoef(
            _ref_flat[_out_mask], _maldi_0_flat[_out_mask]
        )[0, 1]
        corr_he_out_6 = np.corrcoef(
            _ref_flat[_out_mask], _maldi_6_flat[_out_mask]
        )[0, 1]
    else:
        corr_he_out_0, corr_he_out_6 = 0.0, 0.0

    # Correlation between iterations
    if np.sum(_in_mask) > 1:
        corr_0_vs_6_in = np.corrcoef(
            _maldi_0_flat[_in_mask], _maldi_6_flat[_in_mask]
        )[0, 1]
    else:
        corr_0_vs_6_in = 0.0

    if np.sum(_out_mask) > 1:
        corr_0_vs_6_out = np.corrcoef(
            _maldi_0_flat[_out_mask], _maldi_6_flat[_out_mask]
        )[0, 1]
    else:
        corr_0_vs_6_out = 0.0
    return (
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
    )


@app.cell
def _(distance_transform_edt, mask_xr):
    _dist_out = distance_transform_edt(mask_xr.values == 0)
    _dist_in = distance_transform_edt(mask_xr.values == 1)
    dist_to_border = _dist_out - _dist_in
    return (dist_to_border,)


@app.cell
def _(MALDI_transform_0, MALDI_transform_1, REF_pixel_um, dist_to_border):
    _min_dist_px = -15
    _max_dist_px = 40
    _bins = np.arange(_min_dist_px, _max_dist_px + 1)

    # Flatten arrays
    _dists_flat = dist_to_border.flatten()
    _val_0_flat = MALDI_transform_0.values.flatten()
    _val_6_flat = MALDI_transform_1.values.flatten()

    # Digitize distances into bins
    _bin_idx = np.digitize(_dists_flat, _bins) - 1

    _profile_0 = []
    _profile_6 = []
    _std_0 = []
    _std_6 = []
    _dist_centers_um = []

    for _i in range(len(_bins) - 1):
        _in_bin = _bin_idx == _i
        if np.any(_in_bin):
            _profile_0.append(np.mean(_val_0_flat[_in_bin]))
            _profile_6.append(np.mean(_val_6_flat[_in_bin]))
            _std_0.append(np.std(_val_0_flat[_in_bin]))
            _std_6.append(np.std(_val_6_flat[_in_bin]))
        else:
            _profile_0.append(0.0)
            _profile_6.append(0.0)
            _std_0.append(0.0)
            _std_6.append(0.0)
        _dist_centers_um.append((_bins[_i] + 0.5) * REF_pixel_um)

    _max_0 = np.max(_profile_0) if np.max(_profile_0) > 0 else 1.0
    _max_6 = np.max(_profile_6) if np.max(_profile_6) > 0 else 1.0

    radial_profile_df = pl.DataFrame(
        {
            "Distance_um": _dist_centers_um,
            "Iteration_0": _profile_0,
            "Iteration_6": _profile_6,
            "Iteration_0_std": _std_0,
            "Iteration_6_std": _std_6,
            "Iteration_0_norm": np.array(_profile_0) / _max_0,
            "Iteration_6_norm": np.array(_profile_6) / _max_6,
            "Iteration_0_norm_std": np.array(_std_0) / _max_0,
            "Iteration_6_norm_std": np.array(_std_6) / _max_6,
        }
    )
    return (radial_profile_df,)


@app.cell
def _(radial_profile_df):
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
def _(MALDI_transform_0, MALDI_transform_1):
    from skimage.filters import sobel

    _edges_0 = sobel(MALDI_transform_0.values)
    _edges_6 = sobel(MALDI_transform_1.values)

    edges_0_xr = xr.DataArray(
        _edges_0, dims=["Y", "X"], coords=MALDI_transform_0.coords
    )
    edges_6_xr = xr.DataArray(
        _edges_6, dims=["Y", "X"], coords=MALDI_transform_1.coords
    )

    _plot_edges_0 = edges_0_xr.hvplot.image(
        cmap="inferno",
        title="Raw Sobel Edges (Iteration 0)",
        data_aspect=1,
        flip_yaxis=True,
        width=450,
        height=350,
    )
    _plot_edges_6 = edges_6_xr.hvplot.image(
        cmap="inferno",
        title="Deconvolved Sobel Edges (Iteration 6)",
        data_aspect=1,
        flip_yaxis=True,
        width=450,
        height=350,
    )
    sobel_edges_plot = _plot_edges_0 + _plot_edges_6
    return (sobel_edges_plot,)


@app.cell
def _(
    MALDI_transform_0,
    MALDI_transform_1,
    REF_image_transformed_xr,
    REF_pixel_um,
    REF_poly_stream,
    distance_transform_edt,
    mask_xr,
):
    from scipy.ndimage import find_objects, label

    _xs_list = REF_poly_stream.data.get("xs", [])
    _ys_list = REF_poly_stream.data.get("ys", [])
    _valid_polygons = [(xs, ys) for xs, ys in zip(_xs_list, _ys_list) if len(xs) >= 3]
    _y_len, _x_len = mask_xr.shape

    # Get coordinate offsets for proper physical to pixel conversion
    _x_min = float(REF_image_transformed_xr.coords["X"].min())
    _y_min = float(REF_image_transformed_xr.coords["Y"].min())

    widths_0 = []
    widths_6 = []
    cvs_0 = []
    cvs_6 = []

    def _append_roi_metrics(_crop_mask, _patch_0, _patch_6):
        _dist_out = distance_transform_edt(_crop_mask == 0)
        _dist_in = distance_transform_edt(_crop_mask == 1)
        _crop_sdt = _dist_out - _dist_in

        _inside_0 = _patch_0[_crop_mask == 1]
        _inside_6 = _patch_6[_crop_mask == 1]

        if len(_inside_0) > 1 and np.mean(_inside_0) > 0:
            cvs_0.append(np.std(_inside_0) / np.mean(_inside_0))
        if len(_inside_6) > 1 and np.mean(_inside_6) > 0:
            cvs_6.append(np.std(_inside_6) / np.mean(_inside_6))

        _min_dist = -15
        _max_dist = 40
        _bins = np.arange(_min_dist, _max_dist + 1)

        _bin_idx = np.digitize(_crop_sdt.flatten(), _bins) - 1
        _flat_0 = _patch_0.flatten()
        _flat_6 = _patch_6.flatten()

        _p_0 = []
        _p_6 = []
        for _i in range(len(_bins) - 1):
            _in_b = _bin_idx == _i
            _p_0.append(np.mean(_flat_0[_in_b]) if np.any(_in_b) else 0.0)
            _p_6.append(np.mean(_flat_6[_in_b]) if np.any(_in_b) else 0.0)

        def _get_width(_p):
            _p = np.array(_p)
            _mx = np.max(_p)
            if _mx <= 0:
                return None
            _p_norm = _p / _mx
            _ind_90 = np.where(_p_norm <= 0.90)[0]
            _ind_10 = np.where(_p_norm <= 0.10)[0]
            if len(_ind_90) > 0 and len(_ind_10) > 0:
                return _ind_10[0] - _ind_90[0]
            return None

        _w0 = _get_width(_p_0)
        _w6 = _get_width(_p_6)
        if _w0 is not None:
            widths_0.append(_w0 * REF_pixel_um)
        if _w6 is not None:
            widths_6.append(_w6 * REF_pixel_um)

    if len(_valid_polygons) > 0:
        for _xs, _ys in _valid_polygons:
            # Convert physical coordinates (μm) to pixel indices
            # Account for coordinate offset from cropping
            _pts_physical = np.column_stack((_xs, _ys))
            _pts_px = (_pts_physical - np.array([_x_min, _y_min])) / REF_pixel_um
            _px_x = _pts_px[:, 0]
            _px_y = _pts_px[:, 1]

            _x0, _x1 = int(np.floor(np.min(_px_x))), int(np.ceil(np.max(_px_x)))
            _y0, _y1 = int(np.floor(np.min(_px_y))), int(np.ceil(np.max(_px_y)))

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

            _crop_mask_img = Image.new("L", (_crop_w, _crop_h), 0)
            _crop_draw = ImageDraw.Draw(_crop_mask_img)

            _pts_crop = _pts_px - np.array([_x0_pad, _y0_pad])
            _crop_draw.polygon([tuple(_p) for _p in _pts_crop], outline=1, fill=1)
            _crop_mask = np.array(_crop_mask_img)

            _patch_0 = MALDI_transform_0.values[_y0_pad:_y1_pad, _x0_pad:_x1_pad]
            _patch_6 = MALDI_transform_1.values[_y0_pad:_y1_pad, _x0_pad:_x1_pad]
            _append_roi_metrics(_crop_mask, _patch_0, _patch_6)
    else:
        _labeled_mask, _n_components = label(mask_xr.values > 0)
        _component_slices = find_objects(_labeled_mask)
        _min_component_area_px = 25

        for _label_idx, _component_slice in enumerate(_component_slices, start=1):
            if _component_slice is None:
                continue

            _crop_mask = (_labeled_mask[_component_slice] == _label_idx).astype(np.uint8)
            if np.sum(_crop_mask) < _min_component_area_px:
                continue

            _patch_0 = MALDI_transform_0.values[_component_slice]
            _patch_6 = MALDI_transform_1.values[_component_slice]
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
    hvplot.extension("bokeh")

    _dist_um = radial_profile_df["Distance_um"].to_numpy()
    _norm_0 = radial_profile_df["Iteration_0_norm"].to_numpy()
    _norm_6 = radial_profile_df["Iteration_6_norm"].to_numpy()

    def _get_transition_coords(norm_profile):
        _indices_90 = np.where(norm_profile <= 0.90)[0]
        _indices_10 = np.where(norm_profile <= 0.10)[0]
        _idx_90 = _indices_90[0] if len(_indices_90) > 0 else 0
        _idx_10 = (
            _indices_10[0] if len(_indices_10) > 0 else len(norm_profile) - 1
        )
        return _dist_um[_idx_90], _dist_um[_idx_10]

    _t90_0, _t10_0 = _get_transition_coords(_norm_0)
    _t90_6, _t10_6 = _get_transition_coords(_norm_6)

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
    )


@app.cell
def _(
    REF_pixel_um,
    auc_0,
    auc_6,
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
    radial_profile_plot,
    sobel_edges_plot,
    widths_0,
    widths_6,
):
    _display_cv_mean_0 = cv_mean_0 if len(cvs_0) > 0 else cv_in_0
    _display_cv_sd_0 = cv_sd_0 if len(cvs_0) > 1 else 0.0
    _display_cv_mean_6 = cv_mean_6 if len(cvs_6) > 0 else cv_in_6
    _display_cv_sd_6 = cv_sd_6 if len(cvs_6) > 1 else 0.0

    _display_edge_width_mean_0 = edge_width_mean_0 if len(widths_0) > 0 else edge_width_um_0
    _display_edge_width_sd_0 = edge_width_sd_0 if len(widths_0) > 1 else 0.0
    _display_edge_width_mean_6 = edge_width_mean_6 if len(widths_6) > 0 else edge_width_um_6
    _display_edge_width_sd_6 = edge_width_sd_6 if len(widths_6) > 1 else 0.0

    _report = mo.md(
        f"""
        ### Evaluation Metrics (ROI Annotations vs. MALDI)

        | Metric | Iteration 0 (Raw) | Iteration 6 (Deconvolved) | Description |
        |---|---|---|---|
        | **ROC-AUC** | `{auc_0:.4f}` | `{auc_6:.4f}` | Area under the ROC curve (higher is better) |
        | **Intra-ROI CV** | `{_display_cv_mean_0:.4f} ± {_display_cv_sd_0:.4f}` | `{_display_cv_mean_6:.4f} ± {_display_cv_sd_6:.4f}` | Coefficient of variation inside ROIs (lower is smoother) |
        | **Edge Rise Distance** | `{_display_edge_width_mean_0:.2f} ± {_display_edge_width_sd_0:.2f} µm` (`{_display_edge_width_mean_0 / REF_pixel_um:.1f} ± {_display_edge_width_sd_0 / REF_pixel_um:.1f} px`) | `{_display_edge_width_mean_6:.2f} ± {_display_edge_width_sd_6:.2f} µm` (`{_display_edge_width_mean_6 / REF_pixel_um:.1f} ± {_display_edge_width_sd_6 / REF_pixel_um:.1f} px`) | Mean individual ROI edge width (90% to 10% drop, lower is sharper) |
        | *Global Edge Rise* | `{edge_width_um_0:.2f} µm` ({edge_width_px_0} px) | `{edge_width_um_6:.2f} µm` ({edge_width_px_6} px) | Global transition width (90% to 10% drop on average profile) |

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
            mo.md(
                "#### Sobel Gradients (Visualizing Edge Sharpness/Smoothness)"
            ),
            sobel_edges_plot,
        ]
    )
    mo.vstack([_report, _plots])
    return


if __name__ == "__main__":
    app.run()
