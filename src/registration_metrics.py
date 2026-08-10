"""Evaluation metrics comparing registered MALDI data against a reference
ROI annotation, shared by the ``registration`` notebook.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl
from scipy.ndimage import binary_erosion, distance_transform_edt


def cohens_d(inside_values: np.ndarray, outside_values: np.ndarray) -> float:
    """Cohen's d effect size separating two samples (e.g. inside vs.
    outside an ROI mask)."""
    if len(inside_values) < 2 or len(outside_values) < 2:
        return 0.0

    var_in = np.var(inside_values, ddof=1)
    var_out = np.var(outside_values, ddof=1)
    pooled_denom = len(inside_values) + len(outside_values) - 2
    if pooled_denom <= 0:
        return 0.0

    pooled_std = np.sqrt(
        ((len(inside_values) - 1) * var_in + (len(outside_values) - 1) * var_out)
        / pooled_denom
    )
    if pooled_std <= 0:
        return 0.0

    return float((np.mean(inside_values) - np.mean(outside_values)) / pooled_std)


def inside_outside_ratio(inside_values: np.ndarray, outside_values: np.ndarray) -> float:
    """Mean(inside) / Mean(outside), epsilon-guarded against div-by-zero."""
    eps = np.finfo(np.float64).eps
    return float((np.mean(inside_values) + eps) / (np.mean(outside_values) + eps))


def digitize_for_nmi(values: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """Quantile-bin a continuous array into integer labels, for use with
    ``sklearn.metrics.normalized_mutual_info_score``."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or np.allclose(values, values[0]):
        return np.zeros(len(values), dtype=np.int32)

    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros(len(values), dtype=np.int32)

    return np.digitize(values, edges[1:-1], right=False)


def best_threshold_from_pr(
    precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray
) -> Tuple[float, float]:
    """Threshold maximizing F1 along a precision-recall curve.

    Returns (best_threshold, best_f1); (0.0, 0.0) if `thresholds` is empty.
    """
    if len(thresholds) == 0:
        return 0.0, 0.0

    precision_aligned = precision[:-1]
    recall_aligned = recall[:-1]
    f1_scores = np.divide(
        2 * precision_aligned * recall_aligned,
        precision_aligned + recall_aligned,
        out=np.zeros_like(precision_aligned),
        where=(precision_aligned + recall_aligned) > 0,
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """1-pixel-wide boundary of a binary mask (mask XOR its erosion)."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    return np.logical_xor(mask, binary_erosion(mask, border_value=0))


def compute_boundary_metrics(
    pred_values: Optional[np.ndarray], threshold: float, gt_mask: np.ndarray, pixel_size: float,
    boundary_tolerance_px: float = 2,
) -> Tuple[float, float, float, float, float]:
    """Boundary agreement between a thresholded prediction and a ground-
    truth mask: precision/recall/F1 (within `boundary_tolerance_px`),
    mean surface distance, and 95th-percentile (Hausdorff95) surface
    distance, the latter two in physical units via `pixel_size`.

    `pred_values` may be None (e.g. no prediction available), in which case
    all metrics are NaN. Returns
    (precision, recall, f1, mean_surface_distance, hausdorff95), all NaN if
    undetermined.
    """
    gt_boundary = extract_boundary(np.asarray(gt_mask, dtype=bool))

    if pred_values is None or not np.any(gt_boundary):
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    pred_mask = np.asarray(np.asarray(pred_values) >= threshold, dtype=bool)
    pred_boundary = extract_boundary(pred_mask)
    if not np.any(pred_boundary):
        return (0.0, 0.0, 0.0, np.nan, np.nan)

    dist_to_gt = distance_transform_edt(~gt_boundary)
    dist_to_pred = distance_transform_edt(~pred_boundary)

    precision = np.mean(dist_to_gt[pred_boundary] <= boundary_tolerance_px)
    recall = np.mean(dist_to_pred[gt_boundary] <= boundary_tolerance_px)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    pred_to_gt = dist_to_gt[pred_boundary]
    gt_to_pred = dist_to_pred[gt_boundary]
    all_surface_distances = np.concatenate([pred_to_gt, gt_to_pred])

    mean_surface_distance = float(
        0.5 * (pred_to_gt.mean() + gt_to_pred.mean()) * pixel_size
    )
    hausdorff95 = float(np.percentile(all_surface_distances, 95) * pixel_size)

    return (float(precision), float(recall), float(f1), mean_surface_distance, hausdorff95)


def radial_intensity_profile(
    mask: np.ndarray,
    value_arrays: Dict[str, np.ndarray],
    pixel_size: float = 1.0,
    min_dist_px: float = -15,
    max_dist_px: float = 40,
) -> pl.DataFrame:
    """Bin per-pixel intensity by signed distance to a binary mask's border.

    Distance is
    ``distance_transform_edt(mask == 0) - distance_transform_edt(mask == 1)``
    (positive outside the mask, negative inside), digitized into 1-pixel
    bins from `min_dist_px` to `max_dist_px`.

    Parameters
    ----------
    mask:
        Binary mask array; every array in `value_arrays` must match its shape.
    value_arrays:
        ``{name: array}`` of value arrays to profile (e.g. different
        iterations/channels).
    pixel_size:
        Physical size of one pixel; scales the returned bin-center
        distances. Leave at the default ``1.0`` to keep distances in pixels.

    Returns
    -------
    pl.DataFrame with a ``"distance"`` column (bin centers) plus, for each
    key `name` in `value_arrays`, ``f"{name}_mean"`` and ``f"{name}_std"``
    columns (NaN for empty bins).
    """
    mask = np.asarray(mask)
    dist_to_border = distance_transform_edt(mask == 0) - distance_transform_edt(mask == 1)

    bins = np.arange(min_dist_px, max_dist_px + 1)
    bin_idx = np.digitize(dist_to_border.flatten(), bins) - 1

    columns = {"distance": (bins[:-1] + 0.5) * pixel_size}

    for name, values in value_arrays.items():
        flat = np.asarray(values).flatten()
        means = np.full(len(bins) - 1, np.nan)
        stds = np.full(len(bins) - 1, np.nan)
        for i in range(len(bins) - 1):
            in_bin = bin_idx == i
            if np.any(in_bin):
                means[i] = np.mean(flat[in_bin])
                stds[i] = np.std(flat[in_bin])
        columns[f"{name}_mean"] = means
        columns[f"{name}_std"] = stds

    return pl.DataFrame(columns)


def interp_crossing(x0: float, x1: float, y0: float, y1: float, target: float) -> float:
    """Linearly interpolate the x at which y crosses `target`, given two
    bracketing points (x0, y0) and (x1, y1)."""
    if np.isclose(y0, y1):
        return x1
    fraction = (target - y0) / (y1 - y0)
    return x0 + fraction * (x1 - x0)


def _find_threshold_crossing(
    distances: np.ndarray, values: np.ndarray, threshold: float, start_idx: int = 0
) -> Tuple[Optional[float], Optional[int]]:
    """First index (>= start_idx) where `values` drops to/below `threshold`,
    linearly interpolated via `interp_crossing`. Returns
    (crossing_distance, crossing_index), or (None, None) if not found."""
    crossings = np.where(values[start_idx:] <= threshold)[0]
    if len(crossings) == 0:
        return None, None

    idx = start_idx + crossings[0]
    if idx == 0:
        return distances[0], idx

    return (
        interp_crossing(distances[idx - 1], distances[idx], values[idx - 1], values[idx], threshold),
        idx,
    )


def _tail_from_peak(
    distances: np.ndarray, profile: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Restrict to the outward tail starting at the profile's peak at
    distance >= 0 (or the global peak, if the profile never reaches 0)."""
    outside = distances >= 0
    if np.any(outside):
        peak_idx = np.where(outside)[0][np.argmax(profile[outside])]
    else:
        peak_idx = int(np.argmax(profile))
    return distances[peak_idx:], profile[peak_idx:]


def edge_rise_width(
    distances: np.ndarray,
    profile: np.ndarray,
    normalize: str = "self",
    high: float = 0.90,
    low: float = 0.10,
) -> Tuple[Optional[float], Optional[float]]:
    """Distance over which `profile` falls from `high` to `low` of its
    peak, walking outward from the peak found at distance >= 0. The tail is
    made monotonically non-increasing (running minimum) before locating the
    crossings via `interp_crossing`.

    normalize:
        "self" - divide the tail by its own peak value first (for a single
            ROI patch, which has its own intensity scale).
        "none" - assume `profile` is already normalized by the caller (for
            a profile normalized once against the whole image).

    Returns (high_crossing_distance, low_crossing_distance); either may be
    None if it can't be determined (e.g. too few finite samples, or the
    profile never crosses one of the thresholds).
    """
    profile = np.asarray(profile, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    valid = np.isfinite(profile) & np.isfinite(distances)
    profile, distances = profile[valid], distances[valid]
    if len(profile) < 2:
        return None, None

    tail_distances, tail_profile = _tail_from_peak(distances, profile)
    if len(tail_profile) < 2:
        return None, None

    if normalize == "self":
        peak_value = tail_profile[0]
        if peak_value <= 0:
            return None, None
        tail_profile = tail_profile / peak_value
    elif normalize != "none":
        raise ValueError(f"Unknown normalize option: {normalize!r}")

    tail_norm = np.minimum.accumulate(tail_profile)

    high_dist, high_idx = _find_threshold_crossing(tail_distances, tail_norm, high)
    if high_dist is None:
        return None, None
    low_dist, _ = _find_threshold_crossing(tail_distances, tail_norm, low, start_idx=high_idx)
    return high_dist, low_dist


def lsf_fwhm(distances: np.ndarray, profile: np.ndarray) -> float:
    """Full width at half maximum of the line-spread function (LSF),
    derived as ``-d(profile)/d(distance)`` over the same outward tail used
    by `edge_rise_width`.

    `profile` should already be normalized by the caller (as for
    `edge_rise_width(..., normalize="none")`). Returns NaN if it can't be
    determined.
    """
    profile = np.asarray(profile, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    valid = np.isfinite(profile) & np.isfinite(distances)
    profile, distances = profile[valid], distances[valid]
    if len(profile) == 0:
        return np.nan

    tail_distances, tail_profile = _tail_from_peak(distances, profile)
    tail_profile = np.minimum.accumulate(tail_profile)
    if len(tail_distances) < 3:
        return np.nan

    lsf = np.clip(-np.gradient(tail_profile, tail_distances), 0.0, None)
    peak = np.max(lsf)
    if not np.isfinite(peak) or peak <= 0:
        return np.nan

    half_max = 0.5 * peak
    above_half = np.where(lsf >= half_max)[0]
    if len(above_half) == 0:
        return np.nan

    left_idx = int(above_half[0])
    right_idx = int(above_half[-1])

    if left_idx == 0:
        left_cross = tail_distances[0]
    else:
        left_cross = interp_crossing(
            tail_distances[left_idx - 1], tail_distances[left_idx],
            lsf[left_idx - 1], lsf[left_idx], half_max,
        )

    if right_idx == len(lsf) - 1:
        right_cross = tail_distances[-1]
    else:
        right_cross = interp_crossing(
            tail_distances[right_idx], tail_distances[right_idx + 1],
            lsf[right_idx], lsf[right_idx + 1], half_max,
        )

    return float(right_cross - left_cross)
