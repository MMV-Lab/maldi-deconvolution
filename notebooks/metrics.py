import marimo

__generated_with = "0.23.6"
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


    # import sys

    # _root_dir = mo.notebook_dir().parent
    # if str(_root_dir) not in sys.path:
    #     sys.path.append(str(_root_dir))

    from src.psf_generator import MaldiPSFGenerator


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Example Estimation of Automatic / Quantitative Metrics

    This notebook computes three families of metrics to quantify deconvolution quality on MALDI images.

    | Family | Description | Needs |
    |---|---|---|
    | **No-reference metrics** | Single-image statistics (sharpness, sparsity, noise, …) computed independently on raw and deconvolved images | One image per channel |
    | **Comparison metrics** | Cross-image statistics comparing raw vs. deconvolved | Raw + deconvolved pair |
    | **Advanced metrics** | Frequency-domain or forward-model validation (FRC, Decorrelation, Residual Map) | See each section |

    All metrics are computed per m/z channel and then summarised (mean/median) across a user-selected set of channels.
    """)
    return


@app.cell
def _():
    data_xr = xr.open_dataarray(
        r"data/images/SMALDI 1.4/Cerebellum/012-6_Cerebellum_A42_600nm/deconvolved_2026-05-11_08-18-45/Exploris240-2_CC012-6_Cerebellum_DHAP_A42_600nm_900x400_2kV.imzml_deconvolved.h5"
    )

    sim_xr = xr.open_dataset(
        r"data/images/SMALDI 1.4/Cerebellum/012-6_Cerebellum_A42_600nm/deconvolved_2026-05-11_08-18-45/Exploris240-2_CC012-6_Cerebellum_DHAP_A42_600nm_900x400_2kV.imzml_simulation_data.h5"
    )
    return data_xr, sim_xr


@app.cell
def _():
    # Choose which deconvolved RL iteration to evaluate (must be > 0)
    selected_iteration = 10
    # Number of top m/z channels (by median intensity) to include in metric computation
    top_n_mz = 10
    # If not None, bypasses top_n_mz and uses this explicit list of m/z values instead
    selected_mz_list = [834.5923]
    # selected_mz_list = None
    return selected_iteration, selected_mz_list, top_n_mz


@app.function
def compute_metrics(
    da_3d: xr.DataArray, outlier_clip_pct: float | None = None
) -> xr.Dataset:
    """
    Compute 7 no-reference deconvolution quality metrics per m/z channel.

    Parameters
    ----------
    da_3d : xr.DataArray
        Must have dims ``(m_z, y, x)``.
    outlier_clip_pct : float or None
        Per-channel percentile (0–1) at which to clip intensity values before
        computing metrics.  Useful for raw images with saturated pixels.
        Pass ``None`` (default) to skip clipping (suitable for smooth
        deconvolved images).

    Returns
    -------
    xr.Dataset
        One scalar value per metric per m/z value.

    Metrics (expected direction after successful deconvolution)
    -----------------------------------------------------------
    sharpness        ↑  Mean gradient magnitude (Tenenbaum)
    kurtosis         ↑  Excess kurtosis of intensity distribution
    gini             ↑  Gini coefficient (spatial sparsity / localization)
    snr              ↑  Estimated signal-to-noise ratio
    entropy          ↓  Shannon entropy of intensity histogram
    total_variation  ↑  Total variation (sum of |∇I| across the image)
    dynamic_range    ↑  99th-percentile / 5th-percentile ratio (non-zero pixels)
    """

    # Clip per-channel outliers so saturated pixels don't corrupt gradient /
    # range metrics.  clip() broadcasts correctly over m_z.
    if outlier_clip_pct is not None:
        clip_vals = da_3d.quantile(outlier_clip_pct, dim=["y", "x"]).drop_vars(
            "quantile"
        )
        da_3d = da_3d.clip(min=0, max=clip_vals)
    else:
        da_3d = da_3d.clip(min=0)

    # Ensure consistent float64 dtype across all metrics
    da_3d = da_3d.astype(np.float64)

    def _mean_gradient_magnitude(arr: np.ndarray) -> float:
        """Mean gradient magnitude – Tenenbaum sharpness proxy (↑ after deconv)."""
        gy, gx = np.gradient(arr.astype(np.float64))
        return float(np.sqrt(gy**2 + gx**2).mean())

    def _gini_coeff(arr: np.ndarray) -> float:
        """Gini coefficient: spatial concentration of signal (↑ more localized)."""
        flat = np.sort(np.abs(arr.ravel()).astype(np.float64))
        n = len(flat)
        if n == 0 or flat.sum() == 0:
            return 0.0
        idx = np.arange(1, n + 1, dtype=np.float64)
        return float(
            2.0 * (idx * flat).sum() / (n * flat.sum()) - (n + 1.0) / n
        )

    def _shannon_entropy(arr: np.ndarray, n_bins: int = 256) -> float:
        """Shannon entropy of intensity histogram (↓ more structured/ordered)."""
        flat = arr.ravel().astype(np.float64)
        lo, hi = flat.min(), flat.max()
        if hi == lo:
            return 0.0
        counts, _ = np.histogram(flat, bins=n_bins, range=(lo, hi))
        counts = counts[counts > 0].astype(np.float64)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs)))

    def _snr_estimate(arr: np.ndarray) -> float:
        """SNR estimate: mean(above-median) / std(below-median) of non-zero pixels (↑ better)."""
        nonzero = arr.ravel().astype(np.float64)
        nonzero = nonzero[nonzero > 0]
        if len(nonzero) < 4:
            return float("nan")
        thr = np.median(nonzero)
        signal = nonzero[nonzero > thr]
        noise = nonzero[nonzero <= thr]
        if len(signal) == 0 or len(noise) < 2:
            return float("nan")
        noise_std = float(np.std(noise))
        return (
            float(np.mean(signal) / noise_std)
            if noise_std > 0
            else float("nan")
        )

    def _scalar_ufunc(func):
        return xr.apply_ufunc(
            func,
            da_3d,
            input_core_dims=[["y", "x"]],
            vectorize=True,
            output_dtypes=[float],
        )

    # 1. Sharpness – mean gradient magnitude
    sharpness = _scalar_ufunc(_mean_gradient_magnitude)

    # 2. Excess Kurtosis – peakedness of intensity distribution
    mu = da_3d.mean(dim=["y", "x"])
    diff = da_3d - mu
    m2 = (diff**2).mean(dim=["y", "x"])
    m4 = (diff**4).mean(dim=["y", "x"])
    kurtosis = m4 / (m2**2 + 1e-30) - 3.0

    # 3. Gini Coefficient – spatial concentration of signal
    gini = _scalar_ufunc(_gini_coeff)

    # 4. Estimated SNR
    snr = _scalar_ufunc(_snr_estimate)

    # 5. Shannon Entropy of intensity histogram
    entropy = _scalar_ufunc(_shannon_entropy)

    # 6. Total Variation – sum of absolute first-order differences
    tv = da_3d.diff("y").pipe(np.abs).sum(dim=["y", "x"]) + da_3d.diff(
        "x"
    ).pipe(np.abs).sum(dim=["y", "x"])

    # 7. Dynamic Range – 99th-percentile / 5th-percentile (both over non-zero pixels).
    # Using non-zero pixels avoids the median collapsing to 0 on sparse MALDI images,
    # which would make the ratio blow up to ~1e30.
    def _dynamic_range(arr: np.ndarray) -> float:
        nz = arr.ravel().astype(np.float64)
        nz = nz[nz > 0]
        if len(nz) < 2:
            return float("nan")
        p99_val = np.percentile(nz, 99)
        p05_val = np.percentile(nz, 5)
        return float(p99_val / p05_val) if p05_val > 0 else float("nan")

    dynamic_range = _scalar_ufunc(_dynamic_range)

    return xr.Dataset(
        {
            "sharpness": sharpness,
            "kurtosis": kurtosis,
            "gini": gini,
            "snr": snr,
            "total_variation": tv,
            "dynamic_range": dynamic_range,
            "entropy": entropy,
        }
    )


@app.function
def compute_comparison_metrics(
    da_raw: xr.DataArray,
    da_deconv: xr.DataArray,
    da_prev: xr.DataArray | None = None,
) -> xr.Dataset:
    """
    Compute per-m/z cross-comparison metrics between raw and deconvolved DataArrays.
    Both must have dims ``(m_z, y, x)``.

    Metrics
    -------
    pearson_r             Pearson correlation of spatial maps (↑ = structure preserved)
    noise_amplification   Background std ratio deconv/raw (↑ = mottling/over-iteration)
    intensity_conservation  Total intensity ratio deconv/raw (should be ~1.0)
    convergence_delta     Mean |iter_N - iter_{N-1}| per pixel (↓ = converged);
                          NaN when da_prev is None.
    """

    def _pearson_r(arr_raw: np.ndarray, arr_deconv: np.ndarray) -> float:
        r = np.corrcoef(
            arr_raw.ravel().astype(np.float64),
            arr_deconv.ravel().astype(np.float64),
        )[0, 1]
        return float(r)

    def _noise_amplification(
        arr_raw: np.ndarray, arr_deconv: np.ndarray
    ) -> float:
        def _bg_std(arr: np.ndarray) -> float:
            nz = arr.ravel().astype(np.float64)
            nz = nz[nz > 0]
            if len(nz) < 4:
                return 0.0
            bg = nz[nz <= np.median(nz)]  # lower half of non-zero pixels
            return float(np.std(bg)) if len(bg) >= 2 else 0.0

        std_raw = _bg_std(arr_raw)
        return (
            float(_bg_std(arr_deconv) / std_raw)
            if std_raw > 0
            else float("nan")
        )

    def _intensity_conservation(
        arr_raw: np.ndarray, arr_deconv: np.ndarray
    ) -> float:
        total_raw = float(arr_raw.sum())
        return (
            float(arr_deconv.sum() / total_raw)
            if total_raw != 0
            else float("nan")
        )

    def _conv_delta(arr_deconv: np.ndarray, arr_prev: np.ndarray) -> float:
        return float(
            np.abs(
                arr_deconv.astype(np.float64) - arr_prev.astype(np.float64)
            ).mean()
        )

    def _scalar_ufunc2(func):
        return xr.apply_ufunc(
            func,
            da_raw,
            da_deconv,
            input_core_dims=[["y", "x"], ["y", "x"]],
            vectorize=True,
            output_dtypes=[float],
        )

    ds = xr.Dataset(
        {
            "pearson_r": _scalar_ufunc2(_pearson_r),
            "noise_amplification": _scalar_ufunc2(_noise_amplification),
            "intensity_conservation": _scalar_ufunc2(_intensity_conservation),
        }
    )

    if da_prev is not None:
        ds = ds.assign(
            convergence_delta=xr.apply_ufunc(
                _conv_delta,
                da_deconv,
                da_prev,
                input_core_dims=[["y", "x"], ["y", "x"]],
                vectorize=True,
                output_dtypes=[float],
            )
        )
    else:
        ds = ds.assign(
            convergence_delta=xr.full_like(
                ds["pearson_r"], fill_value=float("nan")
            )
        )

    return ds


@app.cell
def _(data_xr, selected_iteration, selected_mz_list, top_n_mz):
    # Select m/z channels: use explicit list if provided, otherwise top N by median intensity
    if selected_mz_list is not None:
        _top_mz = data_xr.m_z.sel(m_z=selected_mz_list)
    else:
        _raw_median = data_xr.sel(iteration=0).median(dim=["y", "x"])
        _top_mz = _raw_median.sortby(_raw_median, ascending=False).m_z[:top_n_mz]

    _da_raw = data_xr.sel(iteration=0, m_z=_top_mz)
    _da_deconv = data_xr.sel(iteration=selected_iteration, m_z=_top_mz)

    _ds_raw = compute_metrics(_da_raw, outlier_clip_pct=0.999)
    _ds_deconv = compute_metrics(_da_deconv)


    def _to_polars(ds: xr.Dataset, label: str) -> pl.DataFrame:
        return pl.from_pandas(ds.to_dataframe().reset_index()).with_columns(
            pl.lit(label).alias("label")
        )


    metrics_df = pl.concat(
        [
            _to_polars(_ds_raw, "raw (iter=0)"),
            _to_polars(_ds_deconv, f"deconvolved (iter={selected_iteration})"),
        ]
    )

    # Convergence delta: compare selected_iteration against the previous stored iteration
    _stored_iters = sorted(data_xr.iteration.values.tolist())
    _prev_idx = _stored_iters.index(selected_iteration) - 1
    _da_prev = (
        data_xr.sel(iteration=_stored_iters[_prev_idx], m_z=_top_mz)
        if _prev_idx >= 0
        else None
    )

    _ds_comparison = compute_comparison_metrics(
        _da_raw, _da_deconv, da_prev=_da_prev
    )
    comparison_df = pl.from_pandas(_ds_comparison.to_dataframe().reset_index())
    return comparison_df, metrics_df


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## No-Reference Metrics

    The following metrics are computed **independently** on each image (no ground-truth or reference required).
    For a well-deconvolved image, all ↑ metrics should increase relative to the raw image and the ↓ metric should decrease.

    | Metric | Formula / Definition | Expected direction | Interpretation |
    |---|---|:---:|---|
    | **sharpness** | Mean gradient magnitude: $\overline{\|\nabla I\|}$ (Tenenbaum) | ↑ | Higher values indicate crisper edges; deconvolution should recover spatial detail blurred by the PSF |
    | **kurtosis** | Excess kurtosis of the pixel intensity distribution: $\mu_4/\sigma^4 - 3$ | ↑ | Peaky distributions (high kurtosis) reflect the sparse, punctate nature of MALDI signal; deconvolution concentrates signal |
    | **gini** | Gini coefficient of the sorted absolute intensities | ↑ | Measures spatial localisation: a value near 1 means most intensity is concentrated in few pixels (sparse signal), near 0 means uniform spread |
    | **snr** | Mean(signal pixels) / Std(background pixels), split at the median | ↑ | Estimated signal-to-noise ratio; deconvolution should improve the separation between signal and noise |
    | **entropy** | Shannon entropy of the 256-bin intensity histogram: $-\sum p_k \log_2 p_k$ | ↓ | Lower entropy means a more structured, less uniformly distributed image; deconvolution reduces noise-driven uniformity |
    | **total_variation** | $\sum_{i,j}(\|I_{i+1,j} - I_{i,j}\| + \|I_{i,j+1} - I_{i,j}\|)$ | ↑ | Sum of all first-order intensity differences; increases when sharp boundaries become more defined |
    | **dynamic_range** | 99th-percentile / 5th-percentile of non-zero pixels | ↑ | Range of the usable intensity scale; deconvolution should amplify peak intensities while suppressing background |
    """)
    return


@app.cell
def _(metrics_df):
    # Per-m/z metrics for raw vs deconvolved
    metrics_df
    return


@app.cell
def _(metrics_df):
    # Median across all m/z channels – single-row summary per method
    metrics_summary_df = (
        metrics_df.group_by("label")
        .agg(pl.selectors.numeric().exclude("m_z").mean())
        .sort("label")
    )
    metrics_summary_df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Comparison Metrics

    These metrics require **both** the raw and deconvolved images and measure how the deconvolution changed the data.

    | Metric | Definition | Expected range | Interpretation |
    |---|---|:---:|---|
    | `pearson_r` | Pearson correlation between the flattened raw and deconvolved spatial maps | 0 to 1 | Values close to 1 confirm that the large-scale spatial structure is preserved; very low values suggest the algorithm diverged or introduced artefacts |
    | `noise_amplification` | Ratio of background std in deconvolved vs. raw image (background = pixels below the image median) | 1 or higher | Values close to 1 are ideal; values >> 1 indicate mottling or noise amplification from over-iteration |
    | `intensity_conservation` | Ratio of total intensity: sum(deconvolved) / sum(raw) | ~1 | Richardson-Lucy conserves total photon count, so this should be close to 1; deviations flag normalisation issues or background subtraction differences |
    | `convergence_delta` | Mean pixel-wise absolute change between iteration N and iteration N-1: $\overline{\|I_N - I_{N-1}\|}$ | decreasing toward 0 | Tracks convergence progress; a very small value signals that the algorithm has effectively stopped updating (NaN when no previous iteration is stored) |
    """)
    return


@app.cell
def _(comparison_df):
    # Cross-comparison metrics (raw vs deconvolved, per m/z)
    comparison_df
    return


@app.cell
def _(comparison_df):
    # Median summary of comparison metrics across top-N m/z channels
    comparison_summary_df = comparison_df.select(
        pl.selectors.numeric().exclude("m_z").median()
    )
    comparison_summary_df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## Advanced Metrics

    Three complementary methods for evaluating deconvolution quality.

    | Method | Goal | Needs |
    |---|---|---|
    | **Fourier Ring Correlation (FRC)** | Quantify resolution recovery in frequency space | Two independent half-datasets per image |
    | **Decorrelation Analysis** | Detect when noise amplification begins | Single image |
    | **Residual Map** | Validate mathematical consistency of the forward model | Raw image + PSF |
    """)
    return


@app.function
def compute_frc_curve(
    img1: np.ndarray,
    img2: np.ndarray,
    pixel_size_um: float = 1.0,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fourier Ring Correlation (FRC) between two 2-D images of the same size.

    For the half-dataset approach, split the source image into odd/even rows
    before calling this function.

    Returns
    -------
    bin_centers : spatial frequencies [1/µm]
    frc         : correlation values ∈ [0, 1]  (NaN for empty bins)
    """
    ny, nx = img1.shape
    F1 = np.fft.fftshift(np.fft.fft2(img1.astype(np.float64)))
    F2 = np.fft.fftshift(np.fft.fft2(img2.astype(np.float64)))

    freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=pixel_size_um))
    freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=pixel_size_um))
    FX, FY = np.meshgrid(freq_x, freq_y)
    R = np.sqrt(FX**2 + FY**2)

    r_max = min(float(np.abs(freq_x).max()), float(np.abs(freq_y).max()))
    bin_edges = np.linspace(0.0, r_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    frc = np.empty(n_bins)
    for i in range(n_bins):
        mask = (R >= bin_edges[i]) & (R < bin_edges[i + 1])
        if mask.sum() == 0:
            frc[i] = np.nan
            continue
        f1v, f2v = F1[mask], F2[mask]
        num = float(np.abs(np.sum(f1v * np.conj(f2v))))
        denom = float(
            np.sqrt(np.sum(np.abs(f1v) ** 2) * np.sum(np.abs(f2v) ** 2))
        )
        frc[i] = num / denom if denom > 0.0 else np.nan

    return bin_centers, frc


@app.function
def frc_resolution_um(
    bin_centers: np.ndarray,
    frc: np.ndarray,
    threshold: float = 0.143,
) -> float:
    """
    Spatial period [µm] where FRC first crosses *threshold*
    (0.143 = 1/7 criterion).  Returns NaN if it never crosses.
    """
    valid = ~np.isnan(frc)
    bc, fc = bin_centers[valid], frc[valid]
    idx = np.where(fc < threshold)[0]
    if len(idx) == 0:
        return float("nan")
    f_cut = bc[idx[0]]
    return float(1.0 / f_cut) if f_cut > 0.0 else float("nan")


@app.function
def compute_halfdata_deconv_frc(
    raw_img: np.ndarray,
    psf_kernel: np.ndarray,
    n_iter: int,
    pixel_size_um: float = 1.0,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Correct half-dataset FRC for deconvolution quality assessment.

    Procedure
    ---------
    1. Split *raw_img* into two independent halves (odd / even rows).
    2. Deconvolve each half **independently** with Richardson-Lucy.
    3. Compute FRC between the two independently deconvolved results.

    Splitting the already-deconvolved image and computing FRC on those halves
    is incorrect: the deterministic RL algorithm applies identical
    high-frequency sharpening to every pixel, so both halves carry perfectly
    correlated high-frequency signal, forcing FRC → 1 everywhere.
    This function avoids that by ensuring each half has its own independent
    noise realisation before deconvolution.

    Returns
    -------
    bin_centers : spatial frequencies [1/µm]
    frc         : correlation values ∈ [0, 1]
    """
    from skimage.restoration import richardson_lucy

    nh = raw_img.shape[0] // 2
    half_a = raw_img[0::2][:nh].astype(np.float64)
    half_b = raw_img[1::2][:nh].astype(np.float64)
    psf = psf_kernel.astype(np.float64)

    def _deconv(half: np.ndarray) -> np.ndarray:
        max_val = float(half.max())
        if max_val <= 0.0:
            return half
        return (
            richardson_lucy(half / max_val, psf, num_iter=n_iter) * max_val
        )

    return compute_frc_curve(
        _deconv(half_a), _deconv(half_b),
        pixel_size_um=pixel_size_um,
        n_bins=n_bins,
    )


@app.function
def compute_decorrelation_curve(
    img: np.ndarray,
    pixel_size_um: float = 1.0,
    n_freqs: int = 60,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simplified decorrelation analysis (after Descloux et al., Nature Methods 2019).

    For each low-pass cutoff frequency f_c the normalised zero-lag
    cross-correlation between the original image and its low-passed version
    is computed:

        d(f_c) = ⟨I · LP(I, f_c)⟩ / (‖I‖_rms · ‖LP(I, f_c)‖_rms)

    Values near 1 indicate real structural content; rapid decay indicates the
    noise floor.  The resolution estimate is the spatial period at which the
    slope of d(f_c) is steepest (maximum of −d′).

    Returns
    -------
    freqs   : spatial frequency samples [1/µm]
    decorr  : normalised correlation values
    res_um  : resolution estimate [µm]  (NaN if not determinable)
    """
    ny, nx = img.shape
    F = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny, d=pixel_size_um))
    freq_x = np.fft.fftshift(np.fft.fftfreq(nx, d=pixel_size_um))
    FX, FY = np.meshgrid(freq_x, freq_y)
    R = np.sqrt(FX**2 + FY**2)

    f_max = min(float(np.abs(freq_x).max()), float(np.abs(freq_y).max()))
    freqs = np.linspace(f_max / n_freqs, f_max, n_freqs)

    I_flat = img.ravel().astype(np.float64)
    rms_I = float(np.sqrt(np.mean(I_flat**2)))
    if rms_I == 0.0:
        return freqs, np.full(n_freqs, np.nan), float("nan")

    decorr = np.empty(n_freqs)
    for i, f_c in enumerate(freqs):
        img_lp = np.real(np.fft.ifft2(np.fft.ifftshift(F * (R <= f_c))))
        lp_flat = img_lp.ravel()
        rms_lp = float(np.sqrt(np.mean(lp_flat**2)))
        decorr[i] = (
            float(np.mean(I_flat * lp_flat) / (rms_I * rms_lp))
            if rms_lp > 0.0
            else np.nan
        )

    # Resolution: spatial period at steepest descent of decorr curve
    valid = ~np.isnan(decorr)
    if valid.sum() < 3:
        return freqs, decorr, float("nan")
    neg_grad = -np.gradient(decorr, freqs)
    neg_grad[~valid] = -np.inf
    peak_f = freqs[int(np.argmax(neg_grad))]
    res_um = float(1.0 / peak_f) if peak_f > 0.0 else float("nan")

    return freqs, decorr, res_um


@app.function
def compute_residual_metrics_per_mz(
    da_raw: xr.DataArray,
    da_deconv: xr.DataArray,
    psf_kernel: np.ndarray,
) -> xr.Dataset:
    """
    Re-blur each deconvolved m/z slice with the PSF and quantify the residual.

        residual = raw - (deconvolved ⊗ PSF)

    An ideal deconvolution produces a residual containing only flat,
    uncorrelated noise.  Structural content in the residual indicates that
    the algorithm fabricated signal or failed to converge.

    Metrics (per m/z)
    -----------------
    residual_std            std(residual)                          ↓ better
    residual_mae            mean |residual|                        ↓ better
    residual_structure_corr |Pearson r| between |residual| and raw
                            Should be ≈ 0 for a clean result       ↓ better
    reblur_pearson_r        Pearson r between raw and re-blurred
                            Measures PSF forward-model consistency  ↑ better
    sse                     Sum of squared errors (raw - reblurred) ↓ better
    rmse                    Root mean squared error                ↓ better
    nrmse_mean              RMSE normalized by data mean           ↓ better
    nrmse_range             RMSE normalized by data range          ↓ better
    nrmse_std               RMSE normalized by data std            ↓ better
    """
    from scipy.signal import fftconvolve

    psf64 = psf_kernel.astype(np.float64)

    def _reblur(arr_deconv: np.ndarray) -> np.ndarray:
        return fftconvolve(arr_deconv.astype(np.float64), psf64, mode="same")

    def _std(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.std(a.astype(np.float64) - _reblur(b)))

    def _mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a.astype(np.float64) - _reblur(b))))

    def _struct(a: np.ndarray, b: np.ndarray) -> float:
        raw64 = a.astype(np.float64)
        residual = raw64 - _reblur(b)
        f1, f2 = raw64.ravel(), np.abs(residual).ravel()
        return (
            float(np.abs(np.corrcoef(f1, f2)[0, 1]))
            if f1.std() > 0 and f2.std() > 0
            else float("nan")
        )

    def _fwd(a: np.ndarray, b: np.ndarray) -> float:
        raw64 = a.astype(np.float64)
        reb = _reblur(b)
        f1, f2 = raw64.ravel(), reb.ravel()
        return (
            float(np.corrcoef(f1, f2)[0, 1])
            if f1.std() > 0 and f2.std() > 0
            else float("nan")
        )

    def _sse(a: np.ndarray, b: np.ndarray) -> float:
        raw64 = a.astype(np.float64)
        residual_sq = (raw64 - _reblur(b)) ** 2
        return float(residual_sq.sum())

    def _rmse(a: np.ndarray, b: np.ndarray) -> float:
        raw64 = a.astype(np.float64)
        residual_sq = (raw64 - _reblur(b)) ** 2
        mse = residual_sq.mean()
        return float(np.sqrt(mse))

    def _nrmse_mean(a: np.ndarray, b: np.ndarray) -> float:
        raw64 = a.astype(np.float64)
        residual_sq = (raw64 - _reblur(b)) ** 2
        mse = residual_sq.mean()
        rmse = np.sqrt(mse)
        data_mean = raw64.mean()
        return float(rmse / data_mean) if data_mean > 0 else float("nan")

    def _nrmse_range(a: np.ndarray, b: np.ndarray) -> float:
        raw64 = a.astype(np.float64)
        residual_sq = (raw64 - _reblur(b)) ** 2
        mse = residual_sq.mean()
        rmse = np.sqrt(mse)
        data_range = raw64.max() - raw64.min()
        return float(rmse / data_range) if data_range > 0 else float("nan")

    def _nrmse_std(a: np.ndarray, b: np.ndarray) -> float:
        raw64 = a.astype(np.float64)
        residual_sq = (raw64 - _reblur(b)) ** 2
        mse = residual_sq.mean()
        rmse = np.sqrt(mse)
        data_std = raw64.std()
        return float(rmse / data_std) if data_std > 0 else float("nan")

    def _uf(func):
        return xr.apply_ufunc(
            func,
            da_raw,
            da_deconv,
            input_core_dims=[["y", "x"], ["y", "x"]],
            vectorize=True,
            output_dtypes=[float],
        )

    return xr.Dataset(
        {
            "residual_std": _uf(_std),
            "residual_mae": _uf(_mae),
            "residual_structure_corr": _uf(_struct),
            "reblur_pearson_r": _uf(_fwd),
            "sse": _uf(_sse),
            "rmse": _uf(_rmse),
            "nrmse_mean": _uf(_nrmse_mean),
            "nrmse_range": _uf(_nrmse_range),
            "nrmse_std": _uf(_nrmse_std),
        }
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 1. Fourier Ring Correlation (FRC)

    For each spatial-frequency ring $R_r$ the FRC is defined as

    $$
    \text{FRC}(r) =
    \frac
      {\left|\displaystyle\sum_{\mathbf{q}\,\in\, R_r} F_1(\mathbf{q})\,\overline{F_2(\mathbf{q})}\right|}
      {\sqrt{\displaystyle\sum_{\mathbf{q}\,\in\, R_r}|F_1(\mathbf{q})|^2
             \cdot\displaystyle\sum_{\mathbf{q}\,\in\, R_r}|F_2(\mathbf{q})|^2}}
    $$

    where $F_1$ and $F_2$ are the DFTs of two **independently** processed
    half-datasets (odd / even rows of the raw image, each deconvolved
    separately).  Resolution is reported at the standard $\text{FRC} = 0.143$
    threshold (1/7 criterion).

    **Output columns** (`frc_res_df`)

    | Column | Description |
    |---|---|
    | `frc_resolution_raw_um` | Spatial period [µm] at which the raw-image FRC curve first crosses the 0.143 threshold — the effective resolution of the raw image |
    | `frc_resolution_raw_px` | Same as above, converted to pixels using the step size |
    | `frc_resolution_deconv_um` | Spatial period [µm] at which the deconvolved-image FRC curve first crosses 0.143 — a smaller value means finer resolved features |
    | `frc_resolution_deconv_px` | Same as above, converted to pixels |

    A decrease in resolution estimate from raw to deconvolved (smaller µm value) confirms that deconvolution has recovered spatial detail beyond the diffraction/PSF limit.
    """)
    return


@app.cell
def _(sim_xr):
    pixel_size_um = float(sim_xr.attrs.get("step_size_um", 1.0))
    pixel_size_um
    return (pixel_size_um,)


@app.cell
def _(
    data_xr,
    pixel_size_um,
    selected_iteration,
    selected_mz_list,
    sim_xr,
    top_n_mz,
):
    if selected_mz_list is not None:
        _top_mz_frc = data_xr.m_z.sel(m_z=selected_mz_list)
    else:
        _raw_med = data_xr.sel(iteration=0).median(dim=["y", "x"])
        _top_mz_frc = _raw_med.sortby(_raw_med, ascending=False).m_z[:top_n_mz]

    _psf_frc = sim_xr.final_signal_kernel.values

    _frc_raw_list, _frc_deconv_list = [], []
    _res_raw_list, _res_deconv_list = [], []
    _bc_ref = None

    for _mz in _top_mz_frc.values:
        _r = data_xr.sel(iteration=0, m_z=_mz).values.astype(np.float64)
        _nh = _r.shape[0] // 2

        # Raw FRC: split raw into odd/even rows and correlate directly
        _bc, _fr = compute_frc_curve(
            _r[0::2][:_nh], _r[1::2][:_nh], pixel_size_um=pixel_size_um
        )
        # Deconvolved FRC: split raw into odd/even rows, deconvolve each half
        # independently, then correlate — the only correct approach.
        # Splitting the already-deconvolved image would give FRC ≈ 1 everywhere
        # because RL applies identical deterministic sharpening to both halves.
        _, _fd = compute_halfdata_deconv_frc(
            _r, _psf_frc, selected_iteration, pixel_size_um=pixel_size_um
        )
        _frc_raw_list.append(_fr)
        _frc_deconv_list.append(_fd)
        _res_raw_list.append(frc_resolution_um(_bc, _fr))
        _res_deconv_list.append(frc_resolution_um(_bc, _fd))
        _bc_ref = _bc

    frc_ds = xr.Dataset(
        {
            "raw": xr.DataArray(
                np.nanmean(_frc_raw_list, axis=0),
                dims=["freq"],
                coords={"freq": _bc_ref},
            ),
            "deconvolved": xr.DataArray(
                np.nanmean(_frc_deconv_list, axis=0),
                dims=["freq"],
                coords={"freq": _bc_ref},
            ),
        }
    )
    frc_ds["freq"].attrs.update(
        {"units": "1/µm", "long_name": "Spatial Frequency"}
    )

    frc_res_df = pl.DataFrame(
        {
            "m_z": [float(v) for v in _top_mz_frc.values],
            "frc_resolution_raw_um": _res_raw_list,
            "frc_resolution_raw_px": [
                r / pixel_size_um if not np.isnan(r) else float("nan")
                for r in _res_raw_list
            ],
            "frc_resolution_deconv_um": _res_deconv_list,
            "frc_resolution_deconv_px": [
                r / pixel_size_um if not np.isnan(r) else float("nan")
                for r in _res_deconv_list
            ],
        }
    )
    return frc_ds, frc_res_df


@app.cell
def _(frc_ds, selected_iteration, selected_mz_list, top_n_mz):
    _n_mz_label = len(selected_mz_list) if selected_mz_list is not None else top_n_mz
    (
        frc_ds.raw.hvplot.line(
            label="raw",
            color="gray",
            line_width=2,
            x="freq",
            xlabel="Spatial Frequency [1/µm]",
            ylabel="FRC",
            title=f"Fourier Ring Correlation — half-dataset, avg. over {_n_mz_label} m/z channels",
            width=700,
            height=400,
        )
        * frc_ds.deconvolved.hvplot.line(
            label=f"deconvolved (iter={selected_iteration})",
            color="steelblue",
            line_width=2,
        )
        * hv.HLine(0.143).opts(color="red", line_dash="dashed", line_width=1.5)
    ).opts(legend_position="top_right", show_grid=True)
    return


@app.cell
def _(frc_res_df):
    frc_res_df
    return


@app.cell
def _(frc_res_df):
    # Mean summary across m/z channels
    frc_summary_df = frc_res_df.select(
        pl.selectors.numeric().exclude("m_z").mean()
    )
    frc_summary_df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 2. Decorrelation Analysis

    For each low-pass cutoff frequency $f_c$ the normalised zero-lag
    cross-correlation between the image $I$ and its low-passed version
    $\mathrm{LP}(I, f_c)$ is

    $$
    d(f_c) =
    \frac{\langle I \cdot \mathrm{LP}(I,\,f_c)\rangle}
         {\|I\|_{\mathrm{rms}}\;\|\mathrm{LP}(I,\,f_c)\|_{\mathrm{rms}}}
    $$

    where $\mathrm{LP}(I,f_c) = \mathcal{F}^{-1}\!\left[\hat{I}(\mathbf{q})\,\mathbf{1}_{|\mathbf{q}|\le f_c}\right]$.
    Values near 1 indicate real structural content; rapid decay marks the
    noise floor.  The resolution estimate $r^*$ is the spatial period at the
    frequency of steepest descent:

    $$r^* = \frac{1}{f_c^*}, \qquad f_c^* = \operatorname*{arg\,max}_{f_c}\left(-\frac{\mathrm{d}\,d}{\mathrm{d}f_c}\right)$$

    **Output columns** (`decorr_res_df`)

    | Column | Description |
    |---|---|
    | `decorr_resolution_raw_um` | Resolution estimate [µm] of the raw image — spatial period at the steepest descent of the decorrelation curve |
    | `decorr_resolution_raw_px` | Same as above, converted to pixels |
    | `decorr_resolution_deconv_um` | Resolution estimate [µm] after deconvolution — should be smaller (finer) than the raw estimate if resolution was improved |
    | `decorr_resolution_deconv_px` | Same as above, converted to pixels |

    Unlike FRC this method works on a **single image** and does not require two independent acquisitions, but it is more sensitive to noise amplification artefacts.
    """)
    return


@app.cell
def _(data_xr, pixel_size_um, selected_iteration, selected_mz_list, top_n_mz):
    if selected_mz_list is not None:
        _top_mz_dc = data_xr.m_z.sel(m_z=selected_mz_list)
    else:
        _raw_med2 = data_xr.sel(iteration=0).median(dim=["y", "x"])
        _top_mz_dc = _raw_med2.sortby(_raw_med2, ascending=False).m_z[:top_n_mz]

    _decorr_raw_list, _decorr_deconv_list = [], []
    _res_raw_dc, _res_deconv_dc = [], []
    _freqs_ref = None

    for _mz in _top_mz_dc.values:
        _r = data_xr.sel(iteration=0, m_z=_mz).values
        _d = data_xr.sel(iteration=selected_iteration, m_z=_mz).values

        _fq, _dr, _rr = compute_decorrelation_curve(
            _r, pixel_size_um=pixel_size_um
        )
        _, _dd, _rd = compute_decorrelation_curve(_d, pixel_size_um=pixel_size_um)
        _decorr_raw_list.append(_dr)
        _decorr_deconv_list.append(_dd)
        _res_raw_dc.append(_rr)
        _res_deconv_dc.append(_rd)
        _freqs_ref = _fq

    decorr_ds = xr.Dataset(
        {
            "raw": xr.DataArray(
                np.nanmean(_decorr_raw_list, axis=0),
                dims=["freq"],
                coords={"freq": _freqs_ref},
            ),
            "deconvolved": xr.DataArray(
                np.nanmean(_decorr_deconv_list, axis=0),
                dims=["freq"],
                coords={"freq": _freqs_ref},
            ),
        }
    )
    decorr_ds["freq"].attrs.update(
        {"units": "1/µm", "long_name": "Spatial Frequency"}
    )

    decorr_res_df = pl.DataFrame(
        {
            "m_z": [float(v) for v in _top_mz_dc.values],
            "decorr_resolution_raw_um": _res_raw_dc,
            "decorr_resolution_raw_px": [
                r / pixel_size_um if not np.isnan(r) else float("nan")
                for r in _res_raw_dc
            ],
            "decorr_resolution_deconv_um": _res_deconv_dc,
            "decorr_resolution_deconv_px": [
                r / pixel_size_um if not np.isnan(r) else float("nan")
                for r in _res_deconv_dc
            ],
        }
    )
    return decorr_ds, decorr_res_df


@app.cell
def _(decorr_ds, selected_iteration, selected_mz_list, top_n_mz):
    _n_mz_label = len(selected_mz_list) if selected_mz_list is not None else top_n_mz
    (
        decorr_ds.raw.hvplot.line(
            label="raw",
            color="gray",
            line_width=2,
            x="freq",
            xlabel="Spatial Frequency [1/µm]",
            ylabel="Normalised Cross-Correlation d(f)",
            title=f"Decorrelation Analysis — avg. over {_n_mz_label} m/z channels",
            width=700,
            height=400,
        )
        * decorr_ds.deconvolved.hvplot.line(
            label=f"deconvolved (iter={selected_iteration})",
            color="steelblue",
            line_width=2,
        )
    ).opts(legend_position="top_right", show_grid=True)
    return


@app.cell
def _(decorr_res_df):
    decorr_res_df
    return


@app.cell
def _(decorr_res_df):
    # Mean summary across m/z channels
    decorr_summary_df = decorr_res_df.select(
        pl.selectors.numeric().exclude("m_z").mean()
    )
    decorr_summary_df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 3. Residual Map (Re-blurring Validation)

    Each deconvolved image is re-convolved with the PSF and compared to the original raw image:

    $$\text{residual} = I_\text{raw} - (I_\text{deconv} \otimes \text{PSF})$$

    An ideal deconvolution produces a residual containing only flat, uncorrelated noise.
    Structural patterns in the residual indicate fabricated signal or failure to converge.

    **Output columns** (`residual_df`) — all computed per m/z channel:

    | Column | Definition | Expected direction | Interpretation |
    |---|---|:---:|---|
    | `residual_std` | Standard deviation of the residual map | lower is better | Measures the overall magnitude of the mismatch between raw and re-blurred; a small std means the forward model is consistent with the data |
    | `residual_mae` | Mean absolute error of the residual: $\overline{\lvert I_\text{raw} - I_\text{reblur} \rvert}$ | lower is better | Robust alternative to std; less sensitive to outlier pixels |
    | `residual_structure_corr` | Absolute Pearson r between $\lvert\text{residual}\rvert$ and the raw image | lower is better | Detects whether the residual correlates spatially with the signal -- a non-zero value means the deconvolution systematically over- or under-estimates signal in high-intensity regions |
    | `reblur_pearson_r` | Pearson r between the raw image and the re-blurred image | higher is better | Validates the PSF forward model; values close to 1 confirm that the stored PSF accurately describes the instrument blur |
    | `sse` | Sum of squared errors: $\sum (I_\text{raw} - I_\text{reblur})^2$ | lower is better | Total energy of the reconstruction error; sensitive to large local deviations |
    | `rmse` | Root mean squared error: $\sqrt{\text{MSE}}$ | lower is better | Scale-aware error magnitude; comparable across iterations |
    | `nrmse_mean` | RMSE divided by the mean of the raw image | lower is better | Normalises RMSE to the signal level, enabling comparison across m/z channels with different intensities |
    | `nrmse_range` | RMSE divided by the intensity range (max - min) of the raw image | lower is better | Normalises by the full dynamic range of the channel |
    | `nrmse_std` | RMSE divided by the standard deviation of the raw image | lower is better | Normalises by the spread of the raw signal distribution |
    """)
    return


@app.cell
def _(data_xr, selected_iteration, selected_mz_list, sim_xr, top_n_mz):
    from scipy.signal import fftconvolve as _fftconvolve

    if selected_mz_list is not None:
        _top_mz_res = data_xr.m_z.sel(m_z=selected_mz_list)
    else:
        _raw_med3 = data_xr.sel(iteration=0).median(dim=["y", "x"])
        _top_mz_res = _raw_med3.sortby(_raw_med3, ascending=False).m_z[:top_n_mz]

    _psf = sim_xr.final_signal_kernel.values

    _da_raw_res = data_xr.sel(iteration=0, m_z=_top_mz_res)
    _da_deconv_res = data_xr.sel(iteration=selected_iteration, m_z=_top_mz_res)

    residual_ds = compute_residual_metrics_per_mz(
        _da_raw_res, _da_deconv_res, _psf
    )
    residual_df = pl.from_pandas(residual_ds.to_dataframe().reset_index())

    # Build residual map for the first m/z in the selection (for visual inspection)
    _top1_mz = float(_top_mz_res.values[0])
    _raw_img = data_xr.sel(iteration=0, m_z=_top1_mz).values.astype(np.float64)
    _dec_img = data_xr.sel(
        iteration=selected_iteration, m_z=_top1_mz
    ).values.astype(np.float64)
    _reb = _fftconvolve(_dec_img, _psf.astype(np.float64), mode="same")

    residual_map_da = xr.DataArray(
        _raw_img - _reb,
        dims=["y", "x"],
        coords={"y": data_xr.y.values, "x": data_xr.x.values},
        name="residual",
        attrs={"long_name": f"Residual (raw − re-blurred), m/z={_top1_mz:.4f}"},
    )
    reblurred_map_da = xr.DataArray(
        _reb,
        dims=["y", "x"],
        coords={"y": data_xr.y.values, "x": data_xr.x.values},
        name="reblurred",
        attrs={"long_name": f"Re-blurred (deconv ⊗ PSF), m/z={_top1_mz:.4f}"},
    )
    raw_map_da = xr.DataArray(
        _raw_img,
        dims=["y", "x"],
        coords={"y": data_xr.y.values, "x": data_xr.x.values},
        name="raw",
        attrs={"long_name": f"Raw, m/z={_top1_mz:.4f}"},
    )
    return raw_map_da, reblurred_map_da, residual_df, residual_map_da


@app.cell
def _(raw_map_da, reblurred_map_da, residual_map_da, selected_iteration):
    _res_log = xr.DataArray(
        np.sign(residual_map_da.values) * np.log1p(np.abs(residual_map_da.values)),
        dims=residual_map_da.dims,
        coords=residual_map_da.coords,
        name="residual_log",
        attrs={"long_name": residual_map_da.attrs.get("long_name", "") + " [signed log1p]"},
    )

    _plot_raw = raw_map_da.hvplot.image(
        title=f"Raw ({raw_map_da.attrs.get('long_name', 'raw')})",
        cmap="magma",
    )
    _plot_reb = reblurred_map_da.hvplot.image(
        title=f"Re-blurred (deconv iter={selected_iteration} ⊗ PSF)",
        cmap="magma",
    )
    _plot_res = _res_log.hvplot.image(
        title="Residual (raw − re-blurred, signed log1p)",
        cmap="RdBu_r",
    )

    hv.Layout([_plot_raw, _plot_reb, _plot_res]).cols(3).opts(
        hv.opts.Image(invert_yaxis=True, frame_height=200, frame_width=200),
        hv.opts.Layout(
            title="Residual Map Analysis (top-1 m/z by median intensity)", 
        ),
    )
    return


@app.cell
def _(residual_df):
    residual_df
    return


@app.cell
def _(residual_df):
    # Mean summary across m/z channels
    residual_summary_df = residual_df.select(
        pl.selectors.numeric().exclude("m_z").mean()
    )
    residual_summary_df
    return


if __name__ == "__main__":
    app.run()
