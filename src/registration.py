"""Image registration, cropping, and masking helpers shared by the
``registration`` notebook.
"""

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import xarray as xr
from PIL import Image, ImageDraw
from scipy.interpolate import splev, splprep
from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_mean,
    threshold_minimum,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)

from .imzml_io import pixel_sizes

# scikit-image global threshold methods available to `mask_from_threshold`.
THRESHOLD_METHODS: Dict[str, Callable[..., float]] = {
    "otsu": threshold_otsu,
    "li": threshold_li,
    "triangle": threshold_triangle,
    "isodata": threshold_isodata,
    "yen": threshold_yen,
    "mean": threshold_mean,
    "minimum": threshold_minimum,
}


def rgb_to_grayscale(
    rgb: xr.DataArray, method: str = "luminosity", channel_dim: str = "S"
) -> xr.DataArray:
    """Convert an RGB xr.DataArray (channels along `channel_dim`, labeled
    "R"/"G"/"B") to a single-channel grayscale xr.DataArray, preserving the
    remaining coordinates/dims.

    method:
        "luminosity" - ITU-R BT.601 weighted sum
            (0.299*R + 0.587*G + 0.114*B). Used for registration and Otsu
            mask thresholding, where the weighting better matches
            perceived intensity.
        "mean" - unweighted average across channels. Used for display
            overlays and correlation/NMI analysis metrics.
    """
    if method == "luminosity":
        r = rgb.sel({channel_dim: "R"})
        g = rgb.sel({channel_dim: "G"})
        b = rgb.sel({channel_dim: "B"})
        return 0.299 * r + 0.587 * g + 0.114 * b
    elif method == "mean":
        return rgb.mean(dim=channel_dim)
    else:
        raise ValueError(f"Unknown method: {method!r}")


def mask_from_threshold(
    gray_image: xr.DataArray, method: str = "otsu", **kwargs
) -> xr.DataArray:
    """Binary mask via a scikit-image global threshold method, preserving
    `gray_image`'s coordinates/dims.

    method:
        One of ``sorted(THRESHOLD_METHODS)`` — currently "otsu" (default),
        "li", "triangle", "isodata", "yen", "mean", "minimum".
    **kwargs:
        Forwarded to the underlying ``skimage.filters.threshold_*``
        function (e.g. ``nbins`` for "otsu"/"isodata"/"yen"/"minimum").
    """
    try:
        threshold_func = THRESHOLD_METHODS[method]
    except KeyError:
        raise ValueError(
            f"Unknown threshold method: {method!r}. Available: {sorted(THRESHOLD_METHODS)}"
        ) from None

    threshold = threshold_func(np.asarray(gray_image), **kwargs)
    return (gray_image > threshold).astype(np.uint8)


def mask_from_polygons(
    polygons: Iterable[Tuple[Sequence[float], Sequence[float]]],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> xr.DataArray:
    """Rasterize physical-coordinate polygons into a binary mask on a given
    coordinate grid.

    Parameters
    ----------
    polygons:
        Iterable of (xs, ys) point-sequence pairs, in the same physical
        units as `x_coords`/`y_coords`.
    x_coords, y_coords:
        1D, evenly-spaced, ascending physical-coordinate arrays defining
        the output grid — typically an existing xr.DataArray's
        ``.coords["X"].values``/``.coords["Y"].values`` (or a slice of
        them), reused directly as the returned DataArray's coordinates so
        it aligns exactly with the source grid.

    Returns
    -------
    xr.DataArray, dims ``("Y", "X")``, coordinates ``X``/``Y``.
    """
    height, width = len(y_coords), len(x_coords)
    x_min, y_min = x_coords[0], y_coords[0]
    pixel_size = x_coords[1] - x_coords[0]

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    for xs, ys in polygons:
        pts_physical = np.column_stack((xs, ys))
        pts_px = (pts_physical - np.array([x_min, y_min])) / pixel_size
        draw.polygon([tuple(p) for p in pts_px], outline=1, fill=1)

    return xr.DataArray(
        np.array(mask_img), dims=("Y", "X"), coords={"X": x_coords, "Y": y_coords}
    )


def smooth_closed_contour(
    x_raw: Sequence[float],
    y_raw: Sequence[float],
    gap_close_threshold: float = 40.0,
    min_points: int = 5,
    smoothing: float = 7,
    n_samples: int = 300,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Snap-close and B-spline-smooth a freehand-drawn closed contour.

    If the start/end points are close but not coincident (within
    `gap_close_threshold`), the loop is snapped shut first. Near-duplicate
    consecutive points (mouse standing still) are dropped to avoid a
    singular fit.

    Returns (x_smooth, y_smooth). If there aren't enough points to
    meaningfully process, returns the input unchanged; if there are too few
    points *after* de-duplication, returns (None, None) to signal the
    caller should drop this contour; if the spline fit itself fails,
    returns the de-duplicated (and possibly closed) raw points.
    """
    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)

    if len(x_raw) < min_points:
        return x_raw, y_raw

    dist_start_end = np.sqrt((x_raw[0] - x_raw[-1]) ** 2 + (y_raw[0] - y_raw[-1]) ** 2)
    if 0 < dist_start_end < gap_close_threshold:
        x_raw = np.append(x_raw, x_raw[0])
        y_raw = np.append(y_raw, y_raw[0])

    keep = np.insert(
        np.sqrt(np.diff(x_raw) ** 2 + np.diff(y_raw) ** 2) > 1e-5, 0, True
    )
    x_raw, y_raw = x_raw[keep], y_raw[keep]

    if len(x_raw) < min_points:
        return None, None

    try:
        tck, _ = splprep([x_raw, y_raw], s=smoothing, per=True)
        u_new = np.linspace(0, 1, n_samples)
        x_smooth, y_smooth = splev(u_new, tck)
        return np.asarray(x_smooth), np.asarray(y_smooth)
    except Exception:
        return x_raw, y_raw


def build_sitk_image(array: np.ndarray, spacing: Tuple[float, float]) -> sitk.Image:
    """Wrap a 2D array as a SimpleITK image with physical pixel spacing."""
    img = sitk.GetImageFromArray(np.asarray(array).astype(np.float32))
    img.SetSpacing(spacing)
    return img


def register_images(
    ref_sitk: sitk.Image,
    moving_sitk: sitk.Image,
    fixed_landmarks: Sequence[Tuple[float, float]],
    moving_landmarks: Sequence[Tuple[float, float]],
    is_multimodal: bool = True,
) -> Tuple[sitk.Transform, float, str]:
    """Landmark-initialized affine registration, refined by SimpleITK
    intensity-based optimization (Mattes mutual information for
    multi-modal pairs such as MALDI vs. H&E, correlation otherwise).

    `fixed_landmarks`/`moving_landmarks` are lists of (x, y) point pairs in
    the same physical units as each image's spacing.

    Returns (transform, metric_value, stop_condition_description).
    """
    fixed_lm = [v for pt in fixed_landmarks for v in (float(pt[0]), float(pt[1]))]
    moving_lm = [v for pt in moving_landmarks for v in (float(pt[0]), float(pt[1]))]

    landmark_initializer = sitk.LandmarkBasedTransformInitializerFilter()
    landmark_initializer.SetFixedLandmarks(fixed_lm)
    landmark_initializer.SetMovingLandmarks(moving_lm)
    initial_transform = landmark_initializer.Execute(sitk.AffineTransform(2))

    reg = sitk.ImageRegistrationMethod()
    # Mattes Mutual Information: standard for multi-modal registration
    if is_multimodal:
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    else:
        reg.SetMetricAsCorrelation()
    reg.SetInterpolator(sitk.sitkNearestNeighbor)

    # Gradient descent with line search
    reg.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-7,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution pyramid (coarse -> fine)
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Start from the landmark-based affine transform
    reg.SetInitialTransform(initial_transform, inPlace=False)

    ref_norm = sitk.RescaleIntensity(sitk.Cast(ref_sitk, sitk.sitkFloat32))
    moving_norm = sitk.RescaleIntensity(sitk.Cast(moving_sitk, sitk.sitkFloat32))

    transform = reg.Execute(ref_norm, moving_norm)

    return transform, reg.GetMetricValue(), reg.GetOptimizerStopConditionDescription()


def apply_transform_resample(
    moving: xr.DataArray, reference_image: sitk.Image, transform: sitk.Transform
) -> np.ndarray:
    """Resample `moving` (an xr.DataArray with µm ``X``/``Y`` coords, from
    which the pixel spacing is derived) onto `reference_image`'s grid using
    `transform`, nearest-neighbor interpolated with a 0.0 fill value.

    Returns a raw ``(y, x)`` array — `reference_image` is a `sitk.Image`
    and doesn't carry usable X/Y coordinate metadata to wrap the result in.
    """
    spacing_y, spacing_x = pixel_sizes(moving)
    moving_sitk = build_sitk_image(moving.values, (spacing_x, spacing_y))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(transform)
    resampled = resampler.Execute(moving_sitk)

    return sitk.GetArrayFromImage(resampled).astype(np.float32)


def crop_to_transformed_bbox(
    transform: sitk.Transform, maldi_image: xr.DataArray, ref_image: xr.DataArray
) -> Tuple[slice, slice]:
    """REF-space (x_slice, y_slice) bounding `maldi_image`'s footprint
    within `ref_image` after registration.

    `transform` maps REF -> MALDI; it is inverted internally to map the
    MALDI image's corner points into REF physical space. Both `maldi_image`
    and `ref_image` must have ``X``/``Y`` coordinate arrays (in µm) to
    slice against; `maldi_image`'s pixel spacing is derived from them.
    """
    try:
        inverse_transform = transform.GetInverse()
    except Exception as e:
        raise RuntimeError(
            "Could not invert the transform to map MALDI coordinates to REF space."
        ) from e

    size_x, size_y = int(maldi_image.sizes["X"]), int(maldi_image.sizes["Y"])
    spacing_y, spacing_x = pixel_sizes(maldi_image)
    corners_local = [
        (0.0, 0.0),
        (size_x * spacing_x, 0.0),
        (0.0, size_y * spacing_y),
        (size_x * spacing_x, size_y * spacing_y),
    ]
    ref_corners = [inverse_transform.TransformPoint(pt) for pt in corners_local]
    ref_xs = [pt[0] for pt in ref_corners]
    ref_ys = [pt[1] for pt in ref_corners]

    min_physical_x, max_physical_x = min(ref_xs), max(ref_xs)
    min_physical_y, max_physical_y = min(ref_ys), max(ref_ys)

    def _slice_bounds(coords: np.ndarray, min_val: float, max_val: float) -> slice:
        if coords[0] < coords[-1]:  # Ascending
            return slice(min_val, max_val)
        else:  # Descending
            return slice(max_val, min_val)

    ref_x_coords = ref_image.coords["X"].values
    ref_y_coords = ref_image.coords["Y"].values

    x_slice = _slice_bounds(ref_x_coords, min_physical_x, max_physical_x)
    y_slice = _slice_bounds(ref_y_coords, min_physical_y, max_physical_y)
    return x_slice, y_slice
