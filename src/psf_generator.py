"""
MALDI PSF Generator
===================

This module provides the `MaldiPSFGenerator` class for generating Point Spread
Functions (PSF) tailored for MALDI-Imaging mass spectrometry, along with the
standalone `check_nyquist_sampling` helper used to validate its sampling
parameters. It accounts for laser beam geometry, sampling step size, raster
scan direction, and signal degradation (decay) due to overlapping past laser
shots.

A `MaldiPSFGenerator` instance runs its full pipeline at construction time: it
builds a fine-resolution simulation grid, accumulates dose from past shots,
converts dose into decay, computes the final per-shot signal, and integrates
everything onto a coarse kernel usable directly as a PSF (e.g. for
deconvolution).

Core Concepts:
--------------
1.  **Dose Map (Fine Grid)**:
    Accumulates exposure from past laser shots at a high resolution
    (`sim_res_um`, 0.05 µm by default).
    - `geometric`: Contains **integer-valued floats** (0.0, 1.0, 2.0...). It represents
      the discrete count of binary overlaps from past shots.
    - `beam_profile`: Contains **continuous floats**. It represents the sum of
      Gaussian intensities from past shots, allowing for subtle degradation modeling.
    "Past" shots are always found assuming a top-left-to-bottom-right raster;
    `scan_direction` ("tl_br" or "bl_tr") then determines whether this map is
    vertically flipped to reflect the actual scan direction.

2.  **Decay Map (Fine Grid)**:
    Transforms the `dose_map` into signal intensity factors (0.0 to 1.0) using
    either exponential fitting or linear interpolation. These are always **continuous floats**.

3.  **Pass Map (Coarse Kernel Grid)**:
    A diagnostic map showing how many past laser shots overlap each coarse pixel.
    - Always contains **integers** (0, 1, 2, 3...).
    - It is a purely geometric count of discrete shots, regardless of the `method` used.
    - Like the dose map, it is computed assuming a top-left-to-bottom-right
      raster and then flipped per `scan_direction`.

4.  **Spatial Profile**:
    The raw beam shape centered at (0,0):
    - `geometric`: A binary circular profile.
    - `beam_profile`: A continuous Gaussian profile.

5.  **Final Signal**:
    The product of the raw `spatial_profile` and the `decay_map`. This represents
    the actual signal density emitted during the current laser shot.

6.  **Kernel Integration (Coarse Grid)**:
    The fine-grid `spatial_profile` and `final_signal` are summed and
    normalized onto the `kernel_size` x `kernel_size` output grid
    (`spatial_kernel`, `final_signal_kernel`) — this coarse, normalized kernel
    is the actual PSF used downstream (see `MaldiPSFGenerator.to_xarray`).

Persistence:
------------
`compute_psf()` is a pure function of the constructor params (no
randomness anywhere in the pipeline, including the `curve_fit` call), so
every attribute above -- fine grids included -- is fully and exactly
re-derivable from those params alone. Rather than writing the fine grids
(tens of MB at typical `sim_res_um`) to disk, use `MaldiPSFGenerator.save`
/ `.load` to round-trip just the params plus the already-small coarse
kernels (a JSON typically well under 1KB); `load_psf_kernels` reads back
just the kernels, for callers that only need the PSF and don't want to
pay for reconstructing the instance.
"""

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
import xarray as xr


def check_nyquist_sampling(
    laser_diameter_um: float, step_size_um: float, oversampling_factor: float = 2.0
) -> bool:
    """Check whether `step_size_um` satisfies the Nyquist-Shannon sampling
    criterion relative to `laser_diameter_um` (the beam's resolution
    element).

    Nyquist requires at least `oversampling_factor` (2, by default) samples
    per resolution element to avoid spatial aliasing:
    ``step_size_um <= laser_diameter_um / oversampling_factor``. Steps
    coarser than this undersample the beam, which both degrades the raw
    image and starves the deconvolution of the overlap it relies on to
    recover resolution.

    Does not raise — prints a warning and returns False when undersampled,
    otherwise returns True.
    """
    nyquist_step_um = laser_diameter_um / oversampling_factor
    is_valid = step_size_um <= nyquist_step_um

    if not is_valid:
        print(
            f"Warning: step_size_um ({step_size_um:.4g} um) exceeds the Nyquist "
            f"limit ({nyquist_step_um:.4g} um = laser_diameter_um / {oversampling_factor:g}) "
            f"for laser_diameter_um={laser_diameter_um:.4g} um. Sampling is coarser "
            f"than the beam's resolution element, risking spatial aliasing and a "
            f"deconvolution with little real overlap to work from. Consider reducing "
            f"step_size_um (or accepting the lower effective resolution)."
        )

    return is_valid


@dataclass
class MaldiPSFGenerator:
    """
    Generates Point Spread Functions (PSF) for MALDI Imaging.

    Simulates the laser beam's spatial profile, the dose accumulated from
    overlapping past shots, and the resulting signal decay, producing both
    a fine-resolution simulation grid and a coarse PSF kernel suitable for
    deconvolution. See the module docstring for the underlying concepts
    (dose map, decay map, pass map, spatial profile, final signal).

    Construction runs the full pipeline immediately: parameters are
    validated (raising `ValueError` on failure), a Nyquist sampling check
    prints a warning (does not raise) if `step_size_um` undersamples the
    beam, and `compute_psf()` populates all derived attributes. There is
    no separate `.fit()` / `.run()` step — the instance is ready to use as
    soon as `__init__` returns.

    Args:
        laser_diameter_um: Laser spot diameter, in µm. Used as the beam's
            1/e^2 diameter (`self.w0 = laser_diameter_um / 2` is the 1/e^2
            radius) for both the Gaussian (`beam_profile`) and binary
            (`geometric`) spatial profiles. Must be > 0.
        step_size_um: Distance between consecutive laser shots, in µm
            (the scan step / pixel pitch). Must be > 0. Checked against
            `laser_diameter_um` via the Nyquist criterion at construction
            time (warns, does not raise, if the step is too coarse).
        kernel_size: Size, in pixels, of the square coarse output kernel
            (`pass_map`, `spatial_kernel`, `final_signal_kernel`). Must be
            a positive odd integer (e.g. 3, 5, 7) so the kernel has a
            well-defined center pixel. Default `3`.
        sim_res_um: Resolution, in µm, of the fine simulation grid used
            internally to compute the dose map and spatial profile before
            integrating down to the coarse kernel. Smaller values are more
            accurate but more expensive. Default `0.05`.
        scan_direction: Laser raster direction: `"tl_br"` (top-left to
            bottom-right) or `"bl_tr"` (bottom-left to top-right).
            Determines which neighboring shots count as "past" shots (and
            so contribute dose/decay) when computing the dose and pass
            maps. Default `"tl_br"`.
        method: Spatial profile model: `"beam_profile"` (continuous
            Gaussian intensity, integer-free) or `"geometric"` (binary
            circular footprint). Affects `spatial_profile`, `dose_map`,
            and `final_signal`; `pass_map` is always a discrete geometric
            count regardless of this setting. Default `"beam_profile"`.
        decay_model: How `dose_map` is converted into `decay_map` values
            in [0, 1]. `"exponential"` fits `a * exp(-b * x) + c` to
            `decay_factors` via `scipy.optimize.curve_fit`, falling back
            to linear interpolation if the fit fails or fewer than 3
            points are given; `"interpolation"` always uses linear
            interpolation directly. Default `"exponential"`.
        decay_factors: Measured normalized signal intensity per number of
            overlapping passes, e.g. `[1.0, 0.8, 0.65]` for 1, 2, 3
            overlapping shots (index `i` corresponds to `i + 1` passes).
            Must be non-empty. Default `[1.0]` (no decay).
        decay_errors: Optional per-point measurement uncertainty (sigma),
            same length as `decay_factors`. When provided, the exponential
            fit uses sigma-weighted least squares (`absolute_sigma=True`)
            for a chi-squared-optimal fit; non-positive values are ignored
            with a warning instead of raising. Default `None`.

    Note:
        The remaining dataclass fields (`xv`, `yv`, `dose_map`,
        `decay_map`, `decay_fit_results`, `spatial_profile`,
        `final_signal`, `spatial_kernel`, `pass_map`,
        `final_signal_kernel`, `fov_span`, `padding`, `meshgrid_span`) are
        computed outputs populated by `compute_psf()` during
        `__post_init__`. They appear as constructor parameters only
        because they are dataclass fields (needed so `compute_psf(**kwargs)`
        can override them programmatically) — they should not normally be
        passed in when creating an instance.
    """

    laser_diameter_um: float
    step_size_um: float
    kernel_size: int = 3
    sim_res_um: float = 0.05
    scan_direction: str = "tl_br"  # 'tl_br' (Top-Left -> Bottom-Right) or 'bl_tr'

    # Configuration overrides
    method: str = "beam_profile"
    decay_model: str = "exponential"
    decay_factors: List[float] = field(default_factory=lambda: [1.0])
    decay_errors: Optional[List[float]] = None  # Per-point sigma; enables weighted fitting when provided

    # Internal storage and computation results
    xv: Optional[np.ndarray] = None
    yv: Optional[np.ndarray] = None
    dose_map: Optional[np.ndarray] = None
    decay_map: Optional[np.ndarray] = None
    decay_fit_results: Optional[dict] = None
    spatial_profile: Optional[np.ndarray] = None
    final_signal: Optional[np.ndarray] = None
    spatial_kernel: Optional[np.ndarray] = None
    pass_map: Optional[np.ndarray] = None
    final_signal_kernel: Optional[np.ndarray] = None

    # Grid parameters
    fov_span: Optional[float] = None
    padding: Optional[float] = None
    meshgrid_span: Optional[float] = None

    # Names of the fields above the "Internal storage and computation
    # results" comment -- i.e. the actual constructor inputs, as opposed to
    # values `compute_psf()` derives from them. Deliberately a plain class
    # attribute (no type annotation) rather than a dataclass field: an
    # annotated one would become a constructor parameter, which isn't what
    # we want for a list of field *names*. Drives `to_params_dict`/`save`.
    _PARAM_FIELDS = (
        "laser_diameter_um",
        "step_size_um",
        "kernel_size",
        "sim_res_um",
        "scan_direction",
        "method",
        "decay_model",
        "decay_factors",
        "decay_errors",
    )

    def __post_init__(self) -> None:
        self._validate()
        self.check_nyquist_sampling()
        self.compute_psf()

    @property
    def w0(self) -> float:
        """1/e^2 beam radius (`laser_diameter_um / 2`)."""
        # Property rather than an attribute set once here, so it can't go
        # stale if laser_diameter_um is later overridden via
        # compute_psf(**kwargs).
        return self.laser_diameter_um / 2.0

    def check_nyquist_sampling(self, oversampling_factor: float = 2.0) -> bool:
        """Warn (without raising) if `step_size_um` undersamples
        `laser_diameter_um` per the Nyquist criterion. See the module-level
        `check_nyquist_sampling` for details; this just forwards this
        instance's parameters to it."""
        return check_nyquist_sampling(
            self.laser_diameter_um, self.step_size_um, oversampling_factor
        )

    def _validate(self) -> None:
        """Raise ValueError on invalid construction parameters."""
        if self.laser_diameter_um <= 0:
            raise ValueError(
                f"laser_diameter_um must be > 0, got {self.laser_diameter_um}."
            )
        if self.step_size_um <= 0:
            raise ValueError(
                f"step_size_um must be > 0, got {self.step_size_um}."
            )
        if not float(self.kernel_size).is_integer():
            raise ValueError(
                f"kernel_size must be a whole number, got {self.kernel_size}."
            )
        if self.kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be a positive integer, got {self.kernel_size}."
            )
        if self.kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size should be odd (e.g. 3, 5), got {self.kernel_size}."
            )
        if self.method not in ("beam_profile", "geometric"):
            raise ValueError(
                f"method must be 'beam_profile' or 'geometric', got {self.method!r}."
            )
        if self.decay_model not in ("exponential", "interpolation"):
            raise ValueError(
                f"decay_model must be 'exponential' or 'interpolation', got {self.decay_model!r}."
            )
        if self.scan_direction not in ("tl_br", "bl_tr"):
            raise ValueError(
                f"scan_direction must be 'tl_br' or 'bl_tr', got {self.scan_direction!r}."
            )
        if not self.decay_factors:
            raise ValueError("decay_factors must be non-empty.")
        if self.decay_errors is not None and len(self.decay_errors) != len(
            self.decay_factors
        ):
            raise ValueError(
                f"decay_errors length ({len(self.decay_errors)}) must match "
                f"decay_factors length ({len(self.decay_factors)})."
            )

    @staticmethod
    def _exponential_decay(
        x: np.ndarray, a: float, b: float, c: float
    ) -> np.ndarray:
        """Exponential decay function: y = a * exp(- b * x) + c"""
        return a * np.exp(-b * x) + c

    def _gaussian_profile(self, r2: np.ndarray) -> np.ndarray:
        """Returns Gaussian intensity for squared radius r2."""
        # I(r) = exp(-2 * r^2 / w0^2). np.exp of a float64 input is
        # already float64 -- no .astype(float) needed (this is the
        # hottest line in the whole pipeline, called ~100+ times per
        # instance over the full fine grid, so the redundant full-array
        # copy wasn't free).
        return np.exp(-2 * r2 / (self.w0**2))

    def _circular_profile(self, r2: np.ndarray) -> np.ndarray:
        """Returns binary circular profile for squared radius r2."""
        # Binary: 1.0 inside radius, 0.0 outside
        return (r2 <= self.w0**2).astype(float)

    def _create_simulation_grid(self) -> None:
        """
        Creates the fine-resolution simulation grid and saves to self.xv, self.yv.
        """
        self.fov_span = self.kernel_size * self.step_size_um
        self.padding = 2 * self.w0
        self.meshgrid_span = self.fov_span / 2.0 + self.padding

        # Grid arrays
        x = np.arange(-self.meshgrid_span, self.meshgrid_span, self.sim_res_um)
        y = np.arange(-self.meshgrid_span, self.meshgrid_span, self.sim_res_um)
        self.xv, self.yv = np.meshgrid(x, y)

        # Current shot spatial profile (method-dependent)
        r_squared = self.xv**2 + self.yv**2
        if self.method == "geometric":
            # Binary circle: 1 inside, 0 outside
            self.spatial_profile = self._circular_profile(r_squared)
        else:  # beam_profile
            # Gaussian distribution
            self.spatial_profile = self._gaussian_profile(r_squared)

    def _compute_dose_map(self) -> None:
        """
        Calculates the accumulated dose/overlap map from previous shots and saves to self.dose_map.
        """
        # Search range
        search_range = (
            int(
                np.ceil(
                    (self.laser_diameter_um / self.step_size_um) * 1.5
                )
            )
            + 2
        )
        self.dose_map = np.zeros_like(self.xv)

        for r in range(-search_range, search_range + 1):
            for c in range(-search_range, search_range + 1):
                # Is this shot in the past? (Assumes TL->BR scan logic initially)
                is_past = False
                if r < 0:
                    is_past = True  # Previous rows
                elif r == 0 and c < 0:
                    is_past = True  # Previous cols in current row

                if not is_past:
                    continue

                # Shot center relative to current (0,0)
                shot_y = r * self.step_size_um
                shot_x = c * self.step_size_um

                dist_sq = (self.xv - shot_x) ** 2 + (self.yv - shot_y) ** 2

                if self.method == "beam_profile":
                    # Accumulated Gaussian intensity
                    self.dose_map += self._gaussian_profile(dist_sq)
                elif self.method == "geometric":
                    # Binary overlap count
                    self.dose_map += self._circular_profile(dist_sq)

        return

    def _apply_decay_model(self) -> None:
        """
        Maps the continuous dose map to decay factors using either interpolation or exponential fitting.
        When decay_errors is provided (3-column CSV input), curve_fit uses sigma weighting
        (absolute_sigma=True) for a chi-squared-optimal fit.
        Updates self.decay_map and self.decay_fit_results.
        """
        x_indices = np.arange(1, len(self.decay_factors) + 1, 1)
        decay_map = None

        # Validate sigma: must match length of decay_factors and contain no zeros
        sigma = None
        if self.decay_errors is not None and len(self.decay_errors) == len(self.decay_factors):
            sigma_arr = np.array(self.decay_errors, dtype=float)
            if np.all(sigma_arr > 0):
                sigma = sigma_arr
            else:
                print("Warning: decay_errors contains zero or negative values; ignoring sigma weighting.")

        # Exponential fit
        if self.decay_model == "exponential" and len(self.decay_factors) >= 3:
            try:
                p0 = [np.max(self.decay_factors), 0.5, np.min(self.decay_factors)]
                fit_kwargs = dict(
                    p0=p0,
                    maxfev=10000,
                )
                if sigma is not None:
                    fit_kwargs["sigma"] = sigma
                    fit_kwargs["absolute_sigma"] = True

                popt, _ = curve_fit(
                    self._exponential_decay,
                    x_indices,
                    self.decay_factors,
                    **fit_kwargs,
                )
                decay_map = self._exponential_decay(self.dose_map, *popt)
                self.decay_fit_results = {
                    "a": popt[0],
                    "b": popt[1],
                    "c": popt[2],
                    "method": "exponential",
                    "weighted": sigma is not None,
                }
                weight_note = " (weighted)" if sigma is not None else ""
                print(
                    f"Exponential fit{weight_note} successful: a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f}"
                )
            except Exception as e:
                print(
                    f"Exponential fit failed ({e}), falling back to linear interpolation."
                )

        # Linear Interpolation fall back
        if decay_map is None:
            decay_func = interp1d(
                x_indices,
                self.decay_factors,
                kind="linear",
                bounds_error=False,
                fill_value=(self.decay_factors[0], self.decay_factors[-1]),
            )
            decay_map = decay_func(self.dose_map)
            self.decay_fit_results = {"method": "interpolation"}

        self.decay_map = decay_map

    def _compute_pass_map(self) -> None:
        """
        Calculates discrete pass count for each kernel pixel.
        Counts how many laser shots (PAST shots only) overlap with each pixel.
        Updates self.pass_map.
        """
        k_size = self.kernel_size
        pass_map = np.zeros((k_size, k_size), dtype=int)
        
        kc = k_size // 2
        step = self.step_size_um
        laser_radius = self.w0  # Using beam radius for overlap check
        
        # Search range for past shots
        search_range = (
            int(np.ceil((self.laser_diameter_um / self.step_size_um) * 1.5)) + 2
        )
        
        for r in range(k_size):
            for c in range(k_size):
                # Physical coordinates of pixel center
                rel_r = r - kc
                rel_c = c - kc
                py = rel_r * step
                px = rel_c * step
                
                # Pixel boundaries
                y_min, y_max = py - step / 2, py + step / 2
                x_min, x_max = px - step / 2, px + step / 2
                
                pass_count = 0
                
                # Check all past shots
                for r_shot in range(-search_range, search_range + 1):
                    for c_shot in range(-search_range, search_range + 1):
                        # Is this shot in the past?
                        is_past = False
                        if r_shot < 0:
                            is_past = True  # Previous rows
                        elif r_shot == 0 and c_shot < 0:
                            is_past = True  # Previous cols in current row
                        
                        if not is_past:
                            continue
                        
                        # Shot center coordinates
                        shot_y = r_shot * step
                        shot_x = c_shot * step
                        
                        # Check if laser circle overlaps with pixel rectangle
                        # Find closest point in pixel to shot center
                        closest_x = np.clip(shot_x, x_min, x_max)
                        closest_y = np.clip(shot_y, y_min, y_max)
                        
                        # Distance from shot center to closest point in pixel
                        dist = np.sqrt((shot_x - closest_x)**2 + (shot_y - closest_y)**2)
                        
                        # If distance <= laser radius, they overlap
                        if dist <= laser_radius:
                            pass_count += 1
                
                pass_map[r, c] = pass_count
        
        self.pass_map = pass_map

    def _integrate_kernel(self) -> None:
        """
        Integrates the fine-grid simulation into the coarse kernel pixels.
        Updates self.spatial_kernel and self.final_signal_kernel.

        Kernel pixel (r, c) covers the half-open square
        `[(r-kc)*step - step/2, (r-kc)*step + step/2)` x (same for c/x) --
        a regular partition of the plane. Rather than rebuilding a
        full-fine-grid boolean mask per kernel pixel (O(kernel_size^2 x
        grid_size), the original approach here), every fine-grid point's
        (row, col) bin is computed once via `floor(v/step + 0.5)` -- the
        closed-form bin index for that same half-open convention -- and
        `np.bincount` scatter-adds directly into the kernel (O(grid_size)).
        Numerically equivalent to the previous per-cell-mask approach
        (verified to agree to float64 noise, ~1e-15, both on step/sim_res
        grids that align exactly on bin boundaries and ones that don't);
        summation order differs, so results aren't bit-identical.
        """
        k_size = self.kernel_size
        kc = k_size // 2
        step = self.step_size_um

        ix = np.floor(self.xv / step + 0.5).astype(np.int64) + kc
        iy = np.floor(self.yv / step + 0.5).astype(np.int64) + kc
        in_bounds = (ix >= 0) & (ix < k_size) & (iy >= 0) & (iy < k_size)

        flat_idx = iy[in_bounds] * k_size + ix[in_bounds]
        spatial_kernel = np.bincount(
            flat_idx, weights=self.spatial_profile[in_bounds], minlength=k_size * k_size
        ).reshape(k_size, k_size)
        final_signal_kernel = np.bincount(
            flat_idx, weights=self.final_signal[in_bounds], minlength=k_size * k_size
        ).reshape(k_size, k_size)

        # Normalize
        if np.sum(final_signal_kernel) > 0:
            final_signal_kernel /= np.sum(final_signal_kernel)
        if np.sum(spatial_kernel) > 0:
            spatial_kernel /= np.sum(spatial_kernel)

        self.spatial_kernel = spatial_kernel
        self.final_signal_kernel = final_signal_kernel

    def to_xarray(self, fov_only: bool = False) -> xr.Dataset:
        """
        Exports the simulation results as an xarray Dataset.
        
        Args:
            fov_only: If True, restricts the grid to the kernel's field of view (fov_span).
                      Otherwise, returns the full meshgrid span including padding.
        """
        y_coords = self.yv[:, 0]
        x_coords = self.xv[0, :]
        
        data_vars = {
            "spatial_profile": (["y", "x"], self.spatial_profile),
            "dose_map": (["y", "x"], self.dose_map),
            "decay_map": (["y", "x"], self.decay_map),
            "final_signal": (["y", "x"], self.final_signal),
        }
        
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={"y": y_coords, "x": x_coords},
        )
        
        ds.x.attrs["units"] = "μm"
        ds.y.attrs["units"] = "μm"
        
        if fov_only and self.fov_span is not None:
            half_fov = self.fov_span / 2.0
            # Slicing the dataset to the requested FOV
            ds = ds.sel(
                x=slice(-half_fov, half_fov),
                y=slice(-half_fov, half_fov)
            )
            
        # Add Kernel versions (Coarse Grid)
        if self.spatial_kernel is not None:
            kc = self.kernel_size // 2
            kx = (np.arange(self.kernel_size) - kc) * self.step_size_um
            ky = (np.arange(self.kernel_size) - kc) * self.step_size_um
            
            ds["spatial_kernel"] = (["y_kernel", "x_kernel"], self.spatial_kernel)
            ds["pass_map"] = (["y_kernel", "x_kernel"], self.pass_map)
            ds["final_signal_kernel"] = (["y_kernel", "x_kernel"], self.final_signal_kernel)
            
            ds = ds.assign_coords({
                "x_kernel": kx,
                "y_kernel": ky
            })
            
            ds.x_kernel.attrs["long_name"] = "x"
            ds.y_kernel.attrs["long_name"] = "y"
            ds.x_kernel.attrs["units"] = "μm"
            ds.y_kernel.attrs["units"] = "μm"

        # Add Decay Data
        if self.decay_factors:
            passes = np.arange(1, len(self.decay_factors) + 1)
            ds["decay_measured"] = (["pass_idx"], self.decay_factors)
            ds = ds.assign_coords({"pass_idx": passes})

            ds.decay_measured.attrs["long_name"] = "Normalized Intensity"
            ds.pass_idx.attrs["long_name"] = "Number of Passes"

            if self.decay_errors is not None and len(self.decay_errors) == len(self.decay_factors):
                ds["decay_errors"] = (["pass_idx"], self.decay_errors)
                ds.decay_errors.attrs["long_name"] = "Measurement Error (sigma)"
            # ds.pass_idx.attrs["units"] = "count"

            # Add Fit Curve if available
            x_fit = np.linspace(0, np.max(passes), 100)
            y_fit = None

            if self.decay_fit_results:
                if self.decay_fit_results.get("method") == "exponential":
                    res = self.decay_fit_results
                    y_fit = self._exponential_decay(x_fit, res["a"], res["b"], res["c"])
                elif self.decay_fit_results.get("method") == "interpolation":
                    # Recreate interpolation function
                    decay_func = interp1d(
                        passes,
                        self.decay_factors,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(self.decay_factors[0], self.decay_factors[-1]),
                    )
                    y_fit = decay_func(x_fit)
                
                if y_fit is not None:
                    ds["decay_fit"] = (["pass_idx_fit"], y_fit)
                    ds = ds.assign_coords({"pass_idx_fit": x_fit})
                    ds.pass_idx_fit.attrs["long_name"] = "Number of Passes"
                    # ds.pass_idx_fit.attrs["units"] = "count"

        ds.attrs.update(
            {
                "laser_diameter_um": self.laser_diameter_um,
                "step_size_um": self.step_size_um,
                "kernel_size": self.kernel_size,
                
                "sim_method": self.method,
                "sim_res_um": self.sim_res_um,
                "scan_direction": self.scan_direction,
                "decay_model": self.decay_model,
                "decay_fit_results": self.decay_fit_results,
                "decay_factors": self.decay_factors,
                "decay_errors": self.decay_errors,
            }
        )


        return ds

    def to_params_dict(self) -> dict:
        """Returns just the constructor inputs (see `_PARAM_FIELDS`) needed
        to reconstruct this instance via `MaldiPSFGenerator(**params)`.
        Excludes every computed attribute (grids, kernels, fit results,
        ...) -- `__post_init__` regenerates those deterministically from
        these inputs alone, so persisting them separately would be
        redundant."""
        return {name: getattr(self, name) for name in self._PARAM_FIELDS}

    def save(self, path: Union[str, Path]) -> None:
        """Persist this instance as a small JSON (typically <1KB) instead
        of dumping the full computed simulation (`to_xarray()`'s fine
        grids run tens of MB at typical `sim_res_um`) to netCDF/HDF5. See
        the module docstring's "Persistence" section.

        Writes the constructor params (`to_params_dict()`) plus the
        already-small coarse kernels/fit results, so downstream code that
        only needs the PSF kernel can read it back via `load_psf_kernels`
        without reconstructing (and re-simulating) a full instance.
        Reload with `MaldiPSFGenerator.load` when the full instance
        (fine grids included) is actually needed again.
        """
        payload = {
            "params": self.to_params_dict(),
            "kernels": {
                "spatial_kernel": self.spatial_kernel.tolist(),
                "pass_map": self.pass_map.tolist(),
                "final_signal_kernel": self.final_signal_kernel.tolist(),
                "decay_fit_results": self.decay_fit_results,
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MaldiPSFGenerator":
        """Reconstruct a `MaldiPSFGenerator` by re-running its full
        pipeline from constructor params previously written by `save`.
        Deterministic, so the result -- fine grids, dose/decay maps,
        kernels, fit results -- exactly reproduces the saved instance;
        this re-simulates rather than reading cached arrays, so prefer
        `load_psf_kernels` if only the PSF kernel itself is needed.
        """
        payload = json.loads(Path(path).read_text())
        return cls(**payload["params"])

    def estimate_kernel_size(self, max_kernel: int = 15) -> None:
        """
        Calculates the percentage of Gaussian beam energy contained within different kernel sizes.
        Prints a report to stdout.
        """

        print("\n--- Kernel Energy Analysis ---")
        print(f"Laser Diameter: {self.laser_diameter_um} um")
        print(f"Step Size:      {self.step_size_um} um")
        print("-" * 45)
        print(f"{'Kernel':<10} | {'Cover (um)':<15} | {'Energy %':<10}")
        print("-" * 45)

        # Reference total energy (using a large grid to approximate infinity)
        large_N = 201
        xs = (np.arange(large_N) - large_N // 2) * self.step_size_um
        ys = (np.arange(large_N) - large_N // 2) * self.step_size_um
        xv, yv = np.meshgrid(xs, ys)
        r2 = xv**2 + yv**2
        total_energy_ref = np.sum(
            self._gaussian_profile(r2)
        )  # Reuse internal profile method

        # Evaluate for requested kernel sizes
        for k in range(3, max_kernel + 1, 2):
            # Create grid for this kernel
            xs_k = (np.arange(k) - k // 2) * self.step_size_um
            ys_k = (np.arange(k) - k // 2) * self.step_size_um
            xv_k, yv_k = np.meshgrid(xs_k, ys_k)

            r2_k = xv_k**2 + yv_k**2
            kernel_energy = np.sum(self._gaussian_profile(r2_k))

            percent = (kernel_energy / total_energy_ref) * 100

            # Physical extent (edge to edge coverage)
            size_um = k * self.step_size_um

            print(f"{k}x{k:<7} | {size_um:<5.1f} x {size_um:<5.1f}   | {percent:.6f}%")

            if percent > 99.99:
                print(f"   >>> {k}x{k} captures >99.99% of signal. Excellent choice.")

    def compute_psf(self, **kwargs) -> None:
        """
        Main orchestrator method to compute PSF.
        Accepts optional overrides for attributes.
        Updates self.spatial_kernel, self.final_signal_kernel, and self.pass_map.
        """
        # Overrides if provided in kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        print(
            f"Generating PSF (Method: {self.method}, Decay: {self.decay_model}, Dir: {self.scan_direction})"
        )
        print(
            f"Beam: {self.laser_diameter_um}um, Step: {self.step_size_um}um"
        )

        # 1. Grid
        self._create_simulation_grid()

        # 2. Dose Map (for decay calculation on fine grid)
        self._compute_dose_map()

        # 3. Pass Map (diagnostic: discrete overlaps on kernel grid)
        self._compute_pass_map()

        # 4. Apply Scan Direction (Vertical Flip if needed)
        if self.scan_direction == "bl_tr":
            self.dose_map = np.flipud(self.dose_map)
            self.pass_map = np.flipud(self.pass_map)

        # 5. Decay
        self._apply_decay_model()

        # 6. Effective Signal
        self.final_signal = self.spatial_profile * self.decay_map

        # 7. Integration
        self._integrate_kernel()


def load_psf_params(path: Union[str, Path]) -> dict:
    """Read just the constructor params (`to_params_dict()`) from a file
    written by `MaldiPSFGenerator.save`, without touching the cached
    kernels or reconstructing an instance. Useful for cheaply recovering
    scalar metadata (e.g. `step_size_um`) that a PSF was generated with.
    """
    payload = json.loads(Path(path).read_text())
    return payload["params"]


def load_psf_kernels(path: Union[str, Path]) -> dict:
    """Read just the small cached kernels from a file written by
    `MaldiPSFGenerator.save`, without reconstructing (and re-simulating)
    the full instance. Use this when only the PSF kernel is needed, e.g.
    for deconvolution or metrics -- see `MaldiPSFGenerator.load` when the
    fine-grid simulation itself is actually needed.

    Returns a dict with keys `spatial_kernel`, `pass_map`,
    `final_signal_kernel` (each `kernel_size` x `kernel_size` ndarrays)
    and `decay_fit_results` (dict).
    """
    payload = json.loads(Path(path).read_text())
    kernels = payload["kernels"]
    return {
        "spatial_kernel": np.array(kernels["spatial_kernel"]),
        "pass_map": np.array(kernels["pass_map"]),
        "final_signal_kernel": np.array(kernels["final_signal_kernel"]),
        "decay_fit_results": kernels["decay_fit_results"],
    }
