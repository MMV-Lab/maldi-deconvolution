import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class PSFConfig:
    """Configuration for MaldiPSFGenerator."""

    laser_diameter_um: float
    step_size_um: float
    kernel_size: int = 3
    sim_res_um: float = 0.05
    scan_direction: str = "tl_br"  # 'tl_br' (Top-Left -> Bottom-Right) or 'bl_tr'


def load_decay_profile(csv_path):
    """
    Loads decay data from CSV and returns a normalized list of decay factors.
    Returns:
        list[float]: Normalized intensity values (1st pass = 1.0).
    """
    if csv_path and os.path.exists(csv_path):
        decay_df = pd.read_csv(csv_path)
        # Sort by pass number just in case
        if "Number of passes" in decay_df.columns:
            decay_df = decay_df.sort_values("Number of passes")

        # Extract intensities
        if "Average intensity" in decay_df.columns:
            intensities = decay_df["Average intensity"].values

            # Normalize relative to the first pass
            if len(intensities) > 0:
                norm_intensities = intensities / intensities[0]
                print("Loaded Signal Decay Data:")
                print(decay_df[["Number of passes", "Average intensity"]].head(10))
                return norm_intensities.tolist()

    print(f"Warning: Decay CSV not found or empty at {csv_path}. Using identity decay.")
    return [1.0]


class MaldiPSFGenerator:
    """
    Generates Point Spread Functions (PSF) for MALDI Imaging.
    Supports modular step-by-step generation.
    """

    def __init__(
        self,
        config: Optional[PSFConfig] = None,
        laser_diameter_um: Optional[float] = None,
        step_size_um: Optional[float] = None,
    ):
        """
        Initialize with a config object OR individual parameters (for validation).
        """
        if config:
            self.config = config
        elif laser_diameter_um is not None and step_size_um is not None:
            self.config = PSFConfig(
                laser_diameter_um=laser_diameter_um, step_size_um=step_size_um
            )
        else:
            raise ValueError(
                "Must provide either 'config' or 'laser_diameter_um' and 'step_size_um'"
            )

        self.w0 = self.config.laser_diameter_um / 2.0  # 1/e^2 radius

    @staticmethod
    def _exponential_decay(x, a, b, c):
        """Exponential decay function: y = a * exp(-b * x) + c"""
        return a * np.exp(-b * x) + c

    def _gaussian_profile(self, r2):
        """Returns Gaussian intensity for squared radius r2."""
        # I(r) = exp(-2 * r^2 / w0^2)
        return np.exp(-2 * r2 / (self.w0**2))

    def _create_simulation_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates the fine-resolution simulation grid.
        Returns:
             xv, yv, beam_spatial_profile
        """
        span = self.config.kernel_size * self.config.step_size_um
        padding = 2 * self.w0
        grid_span = span / 2.0 + padding

        # Grid arrays
        x = np.arange(-grid_span, grid_span, self.config.sim_res_um)
        y = np.arange(-grid_span, grid_span, self.config.sim_res_um)
        xv, yv = np.meshgrid(x, y)

        # Current shot spatial profile (Gaussian centered at 0,0)
        beam_spatial_profile = self._gaussian_profile(xv**2 + yv**2)

        return xv, yv, beam_spatial_profile

    def _compute_dose_map(self, xv, yv, method="beam_profile") -> np.ndarray:
        """
        Calculates the accumulated dose/overlap map from previous shots.
        """
        # Search range
        search_range = (
            int(
                np.ceil(
                    (self.config.laser_diameter_um / self.config.step_size_um) * 1.5
                )
            )
            + 2
        )
        accumulated_dose = np.zeros_like(xv)

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
                shot_y = r * self.config.step_size_um
                shot_x = c * self.config.step_size_um

                dist_sq = (xv - shot_x) ** 2 + (yv - shot_y) ** 2

                if method == "beam_profile":
                    # Exact Gaussian Sum
                    accumulated_dose += self._gaussian_profile(dist_sq)
                elif method == "geometric":
                    # Binary Overlap Count
                    mask = dist_sq <= (self.w0**2)
                    accumulated_dose += mask.astype(float)

        return accumulated_dose

    def _apply_decay_model(
        self,
        dose_map: np.ndarray,
        decay_factors: List[float],
        decay_model="exponential",
    ) -> np.ndarray:
        """
        Maps the continuous dose map to decay factors using either interpolation or exponential fitting.
        """
        x_indices = np.arange(len(decay_factors))
        local_decay_map = None

        # Exponential fit
        if decay_model == "exponential" and len(decay_factors) >= 3:
            try:
                p0 = [1.0, 0.5, 0.0]
                popt, _ = curve_fit(
                    self._exponential_decay,
                    x_indices,
                    decay_factors,
                    p0=p0,
                    maxfev=2000,
                )
                local_decay_map = self._exponential_decay(dose_map, *popt)
                print(
                    f"Exponential fit successful: a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f}"
                )
            except Exception as e:
                print(
                    f"Exponential fit failed ({e}), falling back to linear interpolation."
                )

        # Linear Interpolation fall back
        if local_decay_map is None:
            decay_func = interp1d(
                x_indices,
                decay_factors,
                kind="linear",
                bounds_error=False,
                fill_value=(decay_factors[0], decay_factors[-1]),
            )
            local_decay_map = decay_func(dose_map)

        return local_decay_map

    def _integrate_kernel(
        self, xv, yv, beam_spatial_profile, effective_signal, history_map
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrates the fine-grid simulation into the coarse kernel pixels.
        Returns:
            spatial_kernel, pass_map_kernel, final_psf
        """
        k_size = self.config.kernel_size
        spatial_kernel = np.zeros((k_size, k_size))
        pass_map_kernel = np.zeros((k_size, k_size))
        final_psf = np.zeros((k_size, k_size))

        kc = k_size // 2
        step = self.config.step_size_um

        for r in range(k_size):
            for c in range(k_size):
                rel_r = r - kc
                rel_c = c - kc

                py = rel_r * step
                px = rel_c * step

                y_min, y_max = py - step / 2, py + step / 2
                x_min, x_max = px - step / 2, px + step / 2

                mask = (xv >= x_min) & (xv < x_max) & (yv >= y_min) & (yv < y_max)

                if not np.any(mask):
                    continue

                spatial_kernel[r, c] = np.sum(beam_spatial_profile[mask])
                final_psf[r, c] = np.sum(effective_signal[mask])
                pass_map_kernel[r, c] = np.mean(history_map[mask])

        # Normalize
        if np.sum(final_psf) > 0:
            final_psf /= np.sum(final_psf)
        if np.sum(spatial_kernel) > 0:
            spatial_kernel /= np.sum(spatial_kernel)

        return spatial_kernel, pass_map_kernel, final_psf

    def estimate_kernel_size(self, max_kernel=15):
        """
        Calculates the percentage of Gaussian beam energy contained within different kernel sizes.
        Prints a report to stdout.
        """
        laser = self.config.laser_diameter_um
        step = self.config.step_size_um
        w0 = self.w0  # already calc in init

        print(f"\\n--- Kernel Energy Analysis ---")
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

    def compute_psf(
        self,
        decay_factors=None,
        decay_model="exponential",
        method="beam_profile",
        kernel_size=None,
        scan_direction=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Main orchestrator method to compute PSF.
        Overrides config if parameters are passed.
        """
        # Temp overrides
        if kernel_size:
            self.config.kernel_size = kernel_size
        if scan_direction:
            self.config.scan_direction = scan_direction

        if decay_factors is None:
            decay_factors = [1.0]

        print(
            f"Generating PSF (Method: {method}, Decay: {decay_model}, Dir: {self.config.scan_direction})"
        )
        print(
            f"Beam: {self.config.laser_diameter_um}um, Step: {self.config.step_size_um}um"
        )

        # 1. Grid
        xv, yv, beam_spatial_profile = self._create_simulation_grid()

        # 2. Dose Map
        dose_map = self._compute_dose_map(xv, yv, method=method)

        # 3. Apply Scan Direction (Vertical Flip if needed)
        if self.config.scan_direction == "bl_tr":
            dose_map = np.flipud(dose_map)

        # 4. Decay
        local_decay = self._apply_decay_model(dose_map, decay_factors, decay_model)

        # 5. Effective Signal
        effective_signal = beam_spatial_profile * local_decay

        # 6. Integration
        spatial_kernel, pass_map, final_psf = self._integrate_kernel(
            xv, yv, beam_spatial_profile, effective_signal, dose_map
        )

        return spatial_kernel, pass_map, final_psf
