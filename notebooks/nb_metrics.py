import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    import polars as pl
    import hvplot
    import holoviews as hv
    import hvplot.xarray
    import hvplot.polars
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

    from src.psf_generator import MaldiPSFGenerator, load_psf_kernels, load_psf_params
    from src.imzml_io import load_dataarray
    from src.plotting import image_comparison_layout


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
    data_xr = load_dataarray(
        r"data/images/SMALDI 1.4/Cerebellum/012-6_Cerebellum_A42_600nm/deconvolved_2026-05-11_08-18-45/Exploris240-2_CC012-6_Cerebellum_DHAP_A42_600nm_900x400_2kV.imzml_deconvolved.h5"
    )

    _sim_params_path = r"data/images/SMALDI 1.4/Cerebellum/012-6_Cerebellum_A42_600nm/deconvolved_2026-05-11_08-18-45/Exploris240-2_CC012-6_Cerebellum_DHAP_A42_600nm_900x400_2kV.imzml_simulation_data.json"
    sim_params = load_psf_params(_sim_params_path)
    sim_kernels = load_psf_kernels(_sim_params_path)
    return data_xr, sim_kernels, sim_params


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
        Must have dims ``(M_Z, Y, X)``.
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
        clip_vals = da_3d.quantile(outlier_clip_pct, dim=["Y", "X"]).drop_vars(
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
            input_core_dims=[["Y", "X"]],
            vectorize=True,
            output_dtypes=[float],
        )

    # 1. Sharpness – mean gradient magnitude
    sharpness = _scalar_ufunc(_mean_gradient_magnitude)

    # 2. Excess Kurtosis – peakedness of intensity distribution
    mu = da_3d.mean(dim=["Y", "X"])
    diff = da_3d - mu
    m2 = (diff**2).mean(dim=["Y", "X"])
    m4 = (diff**4).mean(dim=["Y", "X"])
    kurtosis = m4 / (m2**2 + 1e-30) - 3.0

    # 3. Gini Coefficient – spatial concentration of signal
    gini = _scalar_ufunc(_gini_coeff)

    # 4. Estimated SNR
    snr = _scalar_ufunc(_snr_estimate)

    # 5. Shannon Entropy of intensity histogram
    entropy = _scalar_ufunc(_shannon_entropy)

    # 6. Total Variation – sum of absolute first-order differences
    tv = da_3d.diff("Y").pipe(np.abs).sum(dim=["Y", "X"]) + da_3d.diff(
        "X"
    ).pipe(np.abs).sum(dim=["Y", "X"])

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
    Both must have dims ``(M_Z, Y, X)``.

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
            input_core_dims=[["Y", "X"], ["Y", "X"]],
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
                input_core_dims=[["Y", "X"], ["Y", "X"]],
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
        _top_mz = data_xr.M_Z.sel(M_Z=selected_mz_list)
    else:
        _raw_median = data_xr.sel(ITERATION=0).median(dim=["Y", "X"])
        _top_mz = _raw_median.sortby(_raw_median, ascending=False).M_Z[:top_n_mz]

    _da_raw = data_xr.sel(ITERATION=0, M_Z=_top_mz)
    _da_deconv = data_xr.sel(ITERATION=selected_iteration, M_Z=_top_mz)

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
    _stored_iters = sorted(data_xr.ITERATION.values.tolist())
    _prev_idx = _stored_iters.index(selected_iteration) - 1
    _da_prev = (
        data_xr.sel(ITERATION=_stored_iters[_prev_idx], M_Z=_top_mz)
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
            input_core_dims=[["Y", "X"], ["Y", "X"]],
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
def _(sim_params):
    pixel_size_um = float(sim_params.get("step_size_um", 1.0))
    pixel_size_um
    return (pixel_size_um,)


@app.cell
def _(
    data_xr,
    pixel_size_um,
    selected_iteration,
    selected_mz_list,
    sim_kernels,
    top_n_mz,
):
    if selected_mz_list is not None:
        _top_mz_frc = data_xr.M_Z.sel(M_Z=selected_mz_list)
    else:
        _raw_med = data_xr.sel(ITERATION=0).median(dim=["Y", "X"])
        _top_mz_frc = _raw_med.sortby(_raw_med, ascending=False).M_Z[:top_n_mz]

    _psf_frc = sim_kernels["final_signal_kernel"]

    _frc_raw_list, _frc_deconv_list = [], []
    _res_raw_list, _res_deconv_list = [], []
    _bc_ref = None

    for _mz in _top_mz_frc.values:
        _r = data_xr.sel(ITERATION=0, M_Z=_mz).values.astype(np.float64)
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
                dims=["FREQ"],
                coords={"FREQ": _bc_ref},
            ),
            "deconvolved": xr.DataArray(
                np.nanmean(_frc_deconv_list, axis=0),
                dims=["FREQ"],
                coords={"FREQ": _bc_ref},
            ),
        }
    )
    frc_ds["FREQ"].attrs.update(
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
            x="FREQ",
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
        _top_mz_dc = data_xr.M_Z.sel(M_Z=selected_mz_list)
    else:
        _raw_med2 = data_xr.sel(ITERATION=0).median(dim=["Y", "X"])
        _top_mz_dc = _raw_med2.sortby(_raw_med2, ascending=False).M_Z[:top_n_mz]

    _decorr_raw_list, _decorr_deconv_list = [], []
    _res_raw_dc, _res_deconv_dc = [], []
    _freqs_ref = None

    for _mz in _top_mz_dc.values:
        _r = data_xr.sel(ITERATION=0, M_Z=_mz).values
        _d = data_xr.sel(ITERATION=selected_iteration, M_Z=_mz).values

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
                dims=["FREQ"],
                coords={"FREQ": _freqs_ref},
            ),
            "deconvolved": xr.DataArray(
                np.nanmean(_decorr_deconv_list, axis=0),
                dims=["FREQ"],
                coords={"FREQ": _freqs_ref},
            ),
        }
    )
    decorr_ds["FREQ"].attrs.update(
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
            x="FREQ",
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
def _(data_xr, selected_iteration, selected_mz_list, sim_kernels, top_n_mz):
    from scipy.signal import fftconvolve as _fftconvolve

    if selected_mz_list is not None:
        _top_mz_res = data_xr.M_Z.sel(M_Z=selected_mz_list)
    else:
        _raw_med3 = data_xr.sel(ITERATION=0).median(dim=["Y", "X"])
        _top_mz_res = _raw_med3.sortby(_raw_med3, ascending=False).M_Z[:top_n_mz]

    _psf = sim_kernels["final_signal_kernel"]

    _da_raw_res = data_xr.sel(ITERATION=0, M_Z=_top_mz_res)
    _da_deconv_res = data_xr.sel(ITERATION=selected_iteration, M_Z=_top_mz_res)

    residual_ds = compute_residual_metrics_per_mz(
        _da_raw_res, _da_deconv_res, _psf
    )
    residual_df = pl.from_pandas(residual_ds.to_dataframe().reset_index())

    # Build residual map for the first m/z in the selection (for visual inspection)
    _top1_mz = float(_top_mz_res.values[0])
    _raw_img = data_xr.sel(ITERATION=0, M_Z=_top1_mz).values.astype(np.float64)
    _dec_img = data_xr.sel(
        ITERATION=selected_iteration, M_Z=_top1_mz
    ).values.astype(np.float64)
    _reb = _fftconvolve(_dec_img, _psf.astype(np.float64), mode="same")

    residual_map_da = xr.DataArray(
        _raw_img - _reb,
        dims=["Y", "X"],
        coords={"Y": data_xr.Y.values, "X": data_xr.X.values},
        name="residual",
        attrs={"long_name": f"Residual (raw − re-blurred), m/z={_top1_mz:.4f}"},
    )
    reblurred_map_da = xr.DataArray(
        _reb,
        dims=["Y", "X"],
        coords={"Y": data_xr.Y.values, "X": data_xr.X.values},
        name="reblurred",
        attrs={"long_name": f"Re-blurred (deconv ⊗ PSF), m/z={_top1_mz:.4f}"},
    )
    raw_map_da = xr.DataArray(
        _raw_img,
        dims=["Y", "X"],
        coords={"Y": data_xr.Y.values, "X": data_xr.X.values},
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

    image_comparison_layout(
        [_plot_raw, _plot_reb, _plot_res],
        cols=3,
        opts_group="Image",
        invert_yaxis=True,
        frame_height=200,
        frame_width=200,
        title="Residual Map Analysis (top-1 m/z by median intensity)",
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ## Iteration Selection

    To find the best Richardson-Lucy iteration, sweep all stored iterations and track how key metrics
    evolve.  There is no single universally optimal stopping criterion, but the following heuristics
    are reliable:

    | Criterion | What it captures | Expected behaviour |
    |---|---|---|
    | **`grad_cov` increasing-portion elbow** | Coefficient of variation of gradient magnitudes; tracks the *heterogeneity* of edge structure. Has a well-defined minimum at the RL convergence knee, then rises as biological edges develop against a smooth background. The **kneedle elbow of the rising sub-curve** marks where structural-clarity gains flatten out. | Empirically close to visual optimum for denoising-RL on sparse MALDI. |
    | `grad_gini` argmax | Gini coefficient of gradient magnitudes; peaks when gradient energy is concentrated at a few boundaries. | Primary criterion for sharpening-RL. For denoising-RL (scattered-dot images) the raw state already has high Gini, so it decreases monotonically and this branch is skipped. |
    | `convergence_delta` elbow | Mean absolute per-pixel change between consecutive iterations. | Mathematically rigorous convergence indicator; typically 2–4 iterations conservative of the visual optimum. Used as fallback. |
    | All intensity metrics | Pixel-value statistics (sharpness, Gini, SNR, entropy, …). | **Monotone** for RL on sparse MALDI (RL monotonically maximises Poisson log-likelihood) — informative for characterisation, not for stopping. |

    > ⚠️ **Note on statistical robustness:** the `grad_cov` elbow is an *empirical heuristic*
    > grounded in the observation that the coefficient of variation of gradient magnitudes
    > has a concave-increasing profile during the structure-development phase of denoising-RL,
    > and that the kneedle elbow of this profile correlates with the visual quality optimum.
    > It is **not** a mathematically proven optimal stopping criterion.
    > Validate visually on each new dataset, and store iterations at fine spacing
    > (≤1 per step up to at least 2× the expected optimum) to avoid chord-endpoint artefacts
    > in the kneedle detection.

    **Stopping criterion cascade (in order of priority):**
    1. `grad_gini` argmax — fires for sharpening-RL regimes (interior maximum)
    2. Kneedle elbow of `grad_cov` **increasing portion** — fires for denoising-RL regimes (minimum present, elbow past it)
    3. Kneedle elbow of `convergence_delta` — conservative mathematical fallback
    """)
    return


@app.cell
def _(data_xr, selected_mz_list, sim_kernels, top_n_mz):
    from scipy.signal import fftconvolve as _fc_sweep

    _psf_sweep = sim_kernels["final_signal_kernel"]

    _stored_all = sorted(data_xr.ITERATION.values.tolist())
    _nonzero_iters = [i for i in _stored_all if i > 0]

    if selected_mz_list is not None:
        _top_mz_sw = data_xr.M_Z.sel(M_Z=selected_mz_list)
    else:
        _raw_med_sw = data_xr.sel(ITERATION=0).median(dim=["Y", "X"])
        _top_mz_sw = _raw_med_sw.sortby(_raw_med_sw, ascending=False).M_Z[:top_n_mz]

    _da_raw_sw = data_xr.sel(ITERATION=0, M_Z=_top_mz_sw)

    def _nrmse_mean_scalar(arr_raw: np.ndarray, arr_dec: np.ndarray) -> float:
        reb = _fc_sweep(arr_dec.astype(np.float64), _psf_sweep.astype(np.float64), mode="same")
        mse = float(((arr_raw.astype(np.float64) - reb) ** 2).mean())
        data_mean = float(arr_raw.mean())
        return float(np.sqrt(mse) / data_mean) if data_mean > 0 else float("nan")

    def _fixed_bg_noise(arr_raw: np.ndarray, arr_dec: np.ndarray) -> float:
        """Noise leaked into background by RL.

        Background is defined by the raw image: pixels that are exactly zero
        (or in the bottom 5th percentile if no exact zeros exist) should stay
        near zero during correct deconvolution.  When RL over-iterates it
        amplifies noise into these regions, causing this metric to rise.

        Returned value is normalised by the mean raw signal level so it is
        comparable across m/z channels with very different intensities.
        """
        raw_flat = arr_raw.ravel().astype(np.float64)
        dec_flat = arr_dec.ravel().astype(np.float64)

        bg_mask = raw_flat == 0
        if bg_mask.sum() < 4:
            # No exact zeros — fall back to bottom 5th percentile of all pixels
            thr = np.percentile(raw_flat, 5)
            bg_mask = raw_flat <= thr
        if bg_mask.sum() < 4:
            return float("nan")

        signal_mean = float(np.mean(raw_flat[raw_flat > 0])) if (raw_flat > 0).any() else 1.0
        return float(np.std(dec_flat[bg_mask]) / signal_mean) if signal_mean > 0 else float("nan")

    def _grad_gini(arr: np.ndarray) -> float:
        """Gini coefficient of gradient magnitudes.

        Peaks when strong gradients are concentrated at a few structure
        boundaries.  Falls when background haze spreads small gradients
        everywhere (over-iteration) or when the image is blurry (under-iteration).
        """
        gy, gx = np.gradient(arr.astype(np.float64))
        gm = np.sort(np.abs(np.sqrt(gy**2 + gx**2)).ravel())
        n = len(gm)
        if n == 0 or gm.sum() == 0:
            return 0.0
        idx = np.arange(1, n + 1, dtype=np.float64)
        return float(2.0 * (idx * gm).sum() / (n * gm.sum()) - (n + 1.0) / n)

    def _grad_cov(arr: np.ndarray) -> float:
        """Coefficient of variation of gradient magnitudes: std(|∇I|) / mean(|∇I|).

        Behaviour across RL iterations for sparse MALDI data
        ------------------------------------------------------
        Iter 1 (near-raw): isolated bright dots produce moderate local gradients.
          The distribution of |∇I| is moderately heterogeneous → moderate CoV.

        Iter 2-min (convergence phase): RL redistributes signal.  Both std and mean
          of |∇I| fall, but initially at the same relative rate.  The minimum of
          CoV marks the point of *maximum gradient uniformity* — a well-defined
          convergence elbow where the algorithm has completed its initial
          redistribution but biological structure edges are not yet prominent.

        Iter min-N (structure-development phase): as RL builds biological structures,
          the gradient distribution becomes increasingly bimodal — many near-zero
          background gradients and a smaller number of large edge gradients.  Because
          mean(|∇I|) (= sharpness) keeps falling while std(|∇I|) falls more slowly
          (the strong edge pixels are persistent), CoV *rises*.  The rising sub-curve
          is concave: rapid initial rise (structures forming) then flattening
          (diminishing returns in structural clarity).

        Elbow of the rising portion
        ----------------------------
        The kneedle elbow of the concave rising sub-curve identifies where the
        rate of bimodality development drops sharply.  Empirically this corresponds
        to the iteration at which biological structures are well-defined but
        additional iterations produce only marginal improvement in edge contrast.

        IMPORTANT — limitations
        --------------------------
        * This is an *empirical heuristic*, not a mathematically proven optimum.
          Validate visually on each new dataset before trusting it blindly.
        * The elbow position depends on the range of stored iterations: if only
          a few late iterations are stored, the chord used by the kneedle shifts
          and the elbow may move.  Store iterations at fine spacing (≤ 1 per step)
          up to at least 2× the expected optimum.
        * For non-denoising RL regimes (e.g. pure sharpening of continuous images
          with Gaussian noise) the minimum of grad_cov may not appear, and the
          method falls back to the convergence_delta elbow automatically.
        """
        gy, gx = np.gradient(arr.astype(np.float64))
        gm = np.sqrt(gy**2 + gx**2).ravel()
        mu = float(gm.mean())
        return float(gm.std() / mu) if mu > 1e-30 else float("nan")

    _rows = []
    for _it in _nonzero_iters:
        _da_it = data_xr.sel(ITERATION=_it, M_Z=_top_mz_sw)
        _prev_idx_sw = _stored_all.index(_it) - 1
        _da_prev_sw = (
            data_xr.sel(ITERATION=_stored_all[_prev_idx_sw], M_Z=_top_mz_sw)
            if _prev_idx_sw >= 0
            else None
        )
        _ds_nr = compute_metrics(_da_it)
        _ds_cmp = compute_comparison_metrics(_da_raw_sw, _da_it, da_prev=_da_prev_sw)
        _nrmse_vals = xr.apply_ufunc(
            _nrmse_mean_scalar,
            _da_raw_sw,
            _da_it,
            input_core_dims=[["Y", "X"], ["Y", "X"]],
            vectorize=True,
            output_dtypes=[float],
        )
        _fixed_bg_vals = xr.apply_ufunc(
            _fixed_bg_noise,
            _da_raw_sw,
            _da_it,
            input_core_dims=[["Y", "X"], ["Y", "X"]],
            vectorize=True,
            output_dtypes=[float],
        )
        _ufunc_nr = lambda fn: xr.apply_ufunc(
            fn, _da_it,
            input_core_dims=[["Y", "X"]],
            vectorize=True, output_dtypes=[float],
        )
        _rows.append({
            "iteration": _it,
            # gradient-domain stopping criteria (non-monotonic candidates)
            "grad_gini": float(_ufunc_nr(_grad_gini).mean()),
            "grad_cov": float(_ufunc_nr(_grad_cov).mean()),
            # fixed-background noise
            "fixed_bg_noise": float(_fixed_bg_vals.mean()),
            # no-reference
            "sharpness": float(_ds_nr.sharpness.mean()),
            "kurtosis": float(_ds_nr.kurtosis.mean()),
            "gini": float(_ds_nr.gini.mean()),
            "snr": float(_ds_nr.snr.mean()),
            "entropy": float(_ds_nr.entropy.mean()),
            "total_variation": float(_ds_nr.total_variation.mean()),
            "dynamic_range": float(_ds_nr.dynamic_range.mean()),
            # comparison
            "pearson_r": float(_ds_cmp.pearson_r.mean()),
            "noise_amplification": float(_ds_cmp.noise_amplification.mean()),
            "intensity_conservation": float(_ds_cmp.intensity_conservation.mean()),
            "convergence_delta": float(_ds_cmp.convergence_delta.mean()),
            # residual
            "nrmse_mean": float(_nrmse_vals.mean()),
        })

    iteration_sweep_df = pl.DataFrame(_rows)
    iteration_sweep_df
    return (iteration_sweep_df,)


@app.cell
def _(iteration_sweep_df):
    def _find_elbow(iters: np.ndarray, values: np.ndarray) -> int:
        """Kneedle method: index of the point with maximum perpendicular distance
        from the line connecting the first and last observations."""
        x0, y0 = float(iters[0]), float(values[0])
        x1, y1 = float(iters[-1]), float(values[-1])
        dx, dy = x1 - x0, y1 - y0
        norm = np.sqrt(dx**2 + dy**2)
        if norm < 1e-30:
            return 0
        distances = np.abs(dy * (iters.astype(float) - x0) - dx * (values.astype(float) - y0)) / norm
        return int(np.argmax(distances))

    _iters = iteration_sweep_df["iteration"].to_numpy()

    # -----------------------------------------------------------------------
    # Stopping criterion cascade
    # -----------------------------------------------------------------------
    # All single-image intensity metrics (sharpness, SNR, Gini, entropy, …)
    # are monotone for denoising-RL on sparse MALDI because Richardson-Lucy
    # monotonically maximises the Poisson log-likelihood.  The criteria below
    # work in gradient space, which is less directly optimised by RL.
    # -----------------------------------------------------------------------

    # 1. grad_gini argmax ─ primary for "sharpening-RL" regimes
    # ------------------------------------------------------------
    # For data where RL sharpens existing structures (rather than collecting
    # scattered dots), the Gini coefficient of |∇I| peaks at the visual optimum:
    # gradients concentrate at structure edges (high Gini) then spread into noise
    # (falling Gini) with over-iteration.
    # For denoising-RL (scattered-dot MALDI) the raw image already has high
    # gradient Gini (isolated sharp dots), so Gini decreases monotonically →
    # argmax = iter 1 → not an interior point → this branch is skipped.
    _gg = iteration_sweep_df["grad_gini"].to_numpy()
    _gg_best_idx = int(np.nanargmax(_gg))
    _gg_is_interior = 0 < _gg_best_idx < len(_iters) - 1  # excludes boundary artefacts

    # 2. grad_cov increasing-portion elbow ─ primary for denoising-RL regimes
    # -------------------------------------------------------------------------
    # CoV = std(|∇I|) / mean(|∇I|) has a minimum around the convergence knee
    # (transition from redistribution phase to structure-development phase), then
    # rises as the gradient distribution becomes increasingly bimodal (few large
    # edge gradients against many near-zero background gradients).
    #
    # The rising sub-curve is concave: rapid early rise (structures forming) then
    # flattening (diminishing returns).  The kneedle elbow of this sub-curve
    # identifies where further iterations no longer meaningfully improve edge
    # contrast — empirically close to the visual quality optimum.
    #
    # Conditions for this branch to fire:
    #   • grad_cov has a minimum strictly before the last stored iteration
    #   • the increasing portion has ≥ 3 points
    #   • the kneedle selects a point strictly after the minimum
    #     (i.e. not the minimum itself — that is already covered by
    #      convergence_delta elbow)
    _gcov = iteration_sweep_df["grad_cov"].to_numpy()
    _gcov_min_idx = int(np.nanargmin(_gcov))
    _gcov_elbow_iter = None
    if _gcov_min_idx < len(_iters) - 2:
        _incr_iters = _iters[_gcov_min_idx:]   # from minimum to end
        _incr_vals  = _gcov[_gcov_min_idx:]
        if len(_incr_iters) >= 3:
            _incr_elbow_rel = _find_elbow(_incr_iters, _incr_vals)
            if _incr_elbow_rel > 0:             # elbow is past the minimum itself
                _gcov_elbow_iter = int(_incr_iters[_incr_elbow_rel])

    # 3. convergence_delta elbow ─ most conservative fallback
    # ---------------------------------------------------------
    # The mean absolute per-pixel change between consecutive iterations decreases
    # monotonically.  Its kneedle elbow marks where the algorithm stops making
    # substantial progress.  This typically lands 2– 4 iterations *before* the
    # visual optimum for denoising-RL, but is robust to any RL regime.
    _cd_elbow_idx = _find_elbow(_iters, iteration_sweep_df["convergence_delta"].to_numpy())

    if _gg_is_interior:
        best_iteration = int(_iters[_gg_best_idx])
        _criterion = "max grad_gini (sharpening-RL regime)"
    elif _gcov_elbow_iter is not None:
        best_iteration = _gcov_elbow_iter
        _criterion = "elbow of grad_cov increasing portion (denoising-RL regime)"
    else:
        best_iteration = int(_iters[_cd_elbow_idx])
        _criterion = "elbow of convergence_delta (fallback — grad_cov monotone or too few points)"

    _best = iteration_sweep_df.row(
        int(np.where(_iters == best_iteration)[0][0]), named=True
    )
    mo.callout(
        mo.md(
            f"**Suggested best iteration: `{best_iteration}`** ({_criterion})  \n"
            f"grad_gini = {_best['grad_gini']:.4f} — "
            f"grad_cov = {_best['grad_cov']:.4f} — "
            f"convergence_delta = {_best['convergence_delta']:.4f} — "
            f"noise_amplification = {_best['noise_amplification']:.4f}"
        ),
        kind="success",
    )
    return (best_iteration,)


@app.cell
def _(best_iteration, iteration_sweep_df):
    _metrics_to_plot = [
        "grad_gini",
        "grad_cov",
        "convergence_delta",
        "fixed_bg_noise",
        "noise_amplification",
        "nrmse_mean",
        "sharpness",
        "total_variation",
        "kurtosis",
        "dynamic_range",
        "gini",
        "snr",
    ]
    _ylabels = {
        "grad_gini": "Grad Gini (max = best ✓)",
        "grad_cov": "Grad CoV = std/mean (max = best ✓)",
        "convergence_delta": "Convergence delta (elbow ↓)",
        "fixed_bg_noise": "Fixed-BG noise",
        "noise_amplification": "Noise amplification",
        "nrmse_mean": "NRMSE forward model (monotone)",
        "sharpness": "Sharpness (monotone)",
        "total_variation": "Total variation (monotone)",
        "kurtosis": "Kurtosis",
        "dynamic_range": "Dynamic range",
        "gini": "Gini (monotone)",
        "snr": "SNR",
    }
    _plots = []
    for _m in _metrics_to_plot:
        _p = (
            iteration_sweep_df.hvplot.line(
                x="iteration",
                y=_m,
                title=_ylabels[_m],
                width=340,
                height=220,
                line_width=2,
            )
            * iteration_sweep_df.hvplot.scatter(
                x="iteration", y=_m, size=60, color="steelblue"
            )
            * hv.VLine(best_iteration).opts(
                color="red", line_dash="dashed", line_width=1.5
            )
        )
        _plots.append(_p)
    hv.Layout(_plots).cols(3).opts(shared_axes=False)
    return


if __name__ == "__main__":
    app.run()
