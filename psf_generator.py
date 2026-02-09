"""
MALDI PSF Generator
===================

This module provides the MaldiPSFGenerator class for generating Point Spread Functions (PSF)
tailored for MALDI-Imaging mass spectrometry. It accounts for laser beam geometry, 
sampling step size, and signal degradation (decay) due to overlapping past laser shots.

Core Concepts:
--------------
1.  **Dose Map (Fine Grid)**:
    Accumulates exposure from past laser shots at a high resolution (0.05 µm).
    - `geometric`: Contains **integer-valued floats** (0.0, 1.0, 2.0...). It represents 
      the discrete count of binary overlaps from past shots.
    - `beam_profile`: Contains **continuous floats**. It represents the sum of 
      Gaussian intensities from past shots, allowing for subtle degradation modeling.

2.  **Decay Map (Fine Grid)**:
    Transforms the `dose_map` into signal intensity factors (0.0 to 1.0) using 
    either exponential fitting or linear interpolation. These are always **continuous floats**.

3.  **Pass Map (Coarse Kernel Grid)**:
    A diagnostic map showing how many past laser shots overlap each coarse pixel.
    - Always contains **integers** (0, 1, 2, 3...).
    - It is a purely geometric count of discrete shots, regardless of the `method` used.

4.  **Spatial Profile**:
    The raw beam shape centered at (0,0):
    - `geometric`: A binary circular profile.
    - `beam_profile`: A continuous Gaussian profile.

5.  **Final Signal**:
    The product of the raw `spatial_profile` and the `decay_map`. This represents 
    the actual signal density emitted during the current laser shot.
"""

import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import xarray as xr


@dataclass
class MaldiPSFGenerator:
    """
    Generates Point Spread Functions (PSF) for MALDI Imaging.
    Supports modular step-by-step generation.
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

    def __post_init__(self):
        self.w0 = self.laser_diameter_um / 2.0  # 1/e^2 radius
        self.compute_psf()

    @staticmethod
    def _exponential_decay(x, a, b, c):
        """Exponential decay function: y = a * exp(- b * x) + c"""
        return a * np.exp(-b * x) + c

    def _gaussian_profile(self, r2):
        """Returns Gaussian intensity for squared radius r2."""
        # I(r) = exp(-2 * r^2 / w0^2)
        return np.exp(-2 * r2 / (self.w0**2)).astype(float)


    def _circular_profile(self, r2):
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
        Updates self.decay_map and self.decay_fit_results.
        """
        x_indices = np.arange(1, len(self.decay_factors) + 1, 1)
        decay_map = None

        # Exponential fit
        if self.decay_model == "exponential" and len(self.decay_factors) >= 3:
            try:
                p0 = [np.max(self.decay_factors), 0.5, np.min(self.decay_factors)]
                popt, _ = curve_fit(
                    self._exponential_decay,
                    x_indices,
                    self.decay_factors,
                    p0=p0,
                    maxfev=2000,
                )
                decay_map = self._exponential_decay(self.dose_map, *popt)
                self.decay_fit_results = {
                    "a": popt[0],
                    "b": popt[1],
                    "c": popt[2],
                    "method": "exponential",
                }
                print(
                    f"Exponential fit successful: a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f}"
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
        """
        k_size = self.kernel_size
        spatial_kernel = np.zeros((k_size, k_size))
        final_signal_kernel = np.zeros((k_size, k_size))

        kc = k_size // 2
        step = self.step_size_um

        for r in range(k_size):
            for c in range(k_size):
                rel_r = r - kc
                rel_c = c - kc

                py = rel_r * step
                px = rel_c * step

                y_min, y_max = py - step / 2, py + step / 2
                x_min, x_max = px - step / 2, px + step / 2

                mask = (self.xv >= x_min) & (self.xv < x_max) & (self.yv >= y_min) & (self.yv < y_max)

                if not np.any(mask):
                    continue

                spatial_kernel[r, c] = np.sum(self.spatial_profile[mask])
                final_signal_kernel[r, c] = np.sum(self.final_signal[mask])

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

            ds.decay_measured.attrs["long_name"] ="Normalized Intensity"
            ds.pass_idx.attrs["long_name"] = "Number of Passes"
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

        return ds

    def estimate_kernel_size(self, max_kernel=15):
        """
        Calculates the percentage of Gaussian beam energy contained within different kernel sizes.
        Prints a report to stdout.
        """
        laser = self.laser_diameter_um
        step = self.step_size_um
        w0 = self.w0  # already calc in init

        print(f"\n--- Kernel Energy Analysis ---")
        print(f"Laser Diameter: {laser} um")
        print(f"Step Size:      {step} um")
        print("-" * 45)
        print(f"{'Kernel':<10} | {'Cover (um)':<15} | {'Energy %':<10}")
        print("-" * 45)

        # Reference total energy (using a large grid to approximate infinity)
        large_N = 201
        xs = (np.arange(large_N) - large_N // 2) * step
        ys = (np.arange(large_N) - large_N // 2) * step
        xv, yv = np.meshgrid(xs, ys)
        r2 = xv**2 + yv**2
        total_energy_ref = np.sum(
            self._gaussian_profile(r2)
        )  # Reuse internal profile method

        # Evaluate for requested kernel sizes
        for k in range(3, max_kernel + 1, 2):
            # Create grid for this kernel
            xs_k = (np.arange(k) - k // 2) * step
            ys_k = (np.arange(k) - k // 2) * step
            xv_k, yv_k = np.meshgrid(xs_k, ys_k)

            r2_k = xv_k**2 + yv_k**2
            kernel_energy = np.sum(self._gaussian_profile(r2_k))

            percent = (kernel_energy / total_energy_ref) * 100

            # Physical extent (edge to edge coverage)
            size_um = k * step

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
