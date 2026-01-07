"""Build detector response matrices and background spectrum models."""

from __future__ import annotations

from typing import Callable, Dict, Iterable

import numpy as np
from numpy.typing import NDArray

from spectrum.library import Nuclide, NuclideLine

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    _TORCH_AVAILABLE = False

# Electron rest energy.
ME_C2_KEV = 511.0  # keV
# Compton continuum-to-peak ratio (per line).
COMPTON_CONTINUUM_TO_PEAK = 2.0  # Tuned starting value.
# Backscatter peak fraction (tuned default).
BACKSCATTER_FRACTION = 0.03


def gaussian_peak(energy_axis: NDArray[np.float64], center: float, sigma: float) -> NDArray[np.float64]:
    """Return a Gaussian peak with the given center and sigma."""
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    return norm * np.exp(-0.5 * ((energy_axis - center) / sigma) ** 2)


def _resolve_torch_context(
    use_gpu: bool | None,
    gpu_device: str,
    gpu_dtype: str,
) -> tuple["torch.device", "torch.dtype"] | None:
    """Return torch device/dtype when GPU response building is requested and available."""
    if not use_gpu or not _TORCH_AVAILABLE or torch is None:
        return None
    if gpu_device.startswith("cuda") and not torch.cuda.is_available():
        return None
    if gpu_dtype == "float32":
        dtype = torch.float32
    elif gpu_dtype == "float64":
        dtype = torch.float64
    else:
        raise ValueError(f"Unsupported torch dtype: {gpu_dtype}")
    return torch.device(gpu_device), dtype


def _gaussian_peak_torch(
    energy_axis: "torch.Tensor",
    center: float,
    sigma: float,
) -> "torch.Tensor":
    """Return a Gaussian peak on a torch energy axis."""
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    return norm * torch.exp(-0.5 * ((energy_axis - center) / sigma) ** 2)


def _compton_continuum_shape_torch(
    energy_bins_keV: "torch.Tensor",
    E_gamma_keV: float,
    shape: str = "exponential",
) -> "torch.Tensor":
    """Approximate the Compton continuum shape for a single gamma line on torch."""
    Ec = compton_edge_energy(E_gamma_keV)
    continuum = torch.zeros_like(energy_bins_keV)
    if Ec <= 0.0:
        return continuum
    mask = (energy_bins_keV >= 0.0) & (energy_bins_keV <= Ec)
    if not torch.any(mask):
        return continuum
    if shape == "triangular":
        continuum[mask] = energy_bins_keV[mask] / Ec
    elif shape == "exponential":
        tau = Ec / 3.0 if Ec > 0 else 1.0
        continuum[mask] = torch.exp(-energy_bins_keV[mask] / tau)
    else:
        raise ValueError(f"Unknown Compton shape: {shape}")
    total = torch.sum(continuum)
    if float(total) > 0.0:
        continuum = continuum / total
    return continuum


def compton_edge(e_gamma_keV: float) -> float:
    """
    Return the Compton edge energy for a single scatter.

    E_edge = E_gamma * (1 - 1 / (1 + 2 * E_gamma / 511 keV))
    """
    return float(e_gamma_keV * (1.0 - 1.0 / (1.0 + 2.0 * e_gamma_keV / ME_C2_KEV)))


def compton_edge_energy(e_gamma_keV: float) -> float:
    """
    Return the Compton edge energy (keV) of an incident gamma.

    Standard formula using m_e c^2 = 511 keV.
    """
    return compton_edge(e_gamma_keV)


def compton_continuum_shape(
    energy_bins_keV: NDArray[np.float64],
    E_gamma_keV: float,
    shape: str = "exponential",
) -> NDArray[np.float64]:
    """
    Approximate the Compton continuum shape for a single gamma line.

    - Support is [0, Compton edge]
    - shape="exponential" (default) biases low energies; "triangular" is also supported
    """
    E = energy_bins_keV
    Ec = compton_edge_energy(E_gamma_keV)
    mask = (E >= 0.0) & (E <= Ec)
    continuum = np.zeros_like(E, dtype=float)
    if not np.any(mask):
        return continuum
    if shape == "triangular":
        continuum[mask] = E[mask] / Ec
    elif shape == "exponential":
        tau = Ec / 3.0 if Ec > 0 else 1.0
        continuum[mask] = np.exp(-E[mask] / tau)
    else:
        raise ValueError(f"Unknown Compton shape: {shape}")
    total = continuum.sum()
    if total > 0:
        continuum /= total
    return continuum


def backscatter_energy(e_gamma_keV: float) -> float:
    """
    Return the energy after 180-degree backscatter.

    E_back = E_gamma / (1 + 2 E_gamma / 511 keV)
    """
    return float(e_gamma_keV / (1.0 + 2.0 * e_gamma_keV / ME_C2_KEV))


def compton_continuum(
    energy_axis: NDArray[np.float64],
    e_gamma_keV: float,
    bin_width_keV: float,
    peak_area: float,
    continuum_to_peak: float = COMPTON_CONTINUUM_TO_PEAK,
    shape_power: float = 2.0,
) -> NDArray[np.float64]:
    """
    Generate a Compton continuum contribution for a single gamma line.

    - Non-zero only for 0 < E < Compton edge
    - Monotonically decreasing toward higher energy
    - Normalised so total area equals continuum_to_peak * peak_area
    """
    if peak_area <= 0.0:
        return np.zeros_like(energy_axis, dtype=float)
    base = compton_continuum_shape(energy_axis, e_gamma_keV, shape="exponential")
    norm = base.sum() * bin_width_keV
    if norm <= 0:
        return np.zeros_like(energy_axis, dtype=float)
    scale = (continuum_to_peak * peak_area) / norm
    return base * scale


def default_background_shape(energy_axis_keV: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Return a CeBr3-like background shape (normalised).

    - Gentle bump around 100 keV
    - Exponential decay thereafter
    """
    E = np.asarray(energy_axis_keV, dtype=float)
    bump = np.exp(-0.5 * ((E - 100.0) / 50.0) ** 2)
    decay = np.exp(-E / 400.0)
    bg = 0.4 * bump + decay
    bg[E < 30.0] = 0.0
    # Normalise to unit sum; bin width is handled by the caller.
    total = bg.sum()
    if total > 0:
        bg = bg / total
    return bg


def default_resolution() -> Callable[[float], float]:
    """
    Return a CeBr3-like energy resolution sigma(E) function.

    Based on a Scionix application note: ~8% at 122 keV, ~4% at 662 keV, and ~3% at
    1332 keV. We use sigma(E) = max(0.5 * sqrt(E) - 1.5, 0.1) (FWHM = 2.355*sigma).
    """

    def sigma(energy_keV: float) -> float:
        return max(0.5 * np.sqrt(energy_keV) - 1.5, 0.1)

    return sigma


def constant_efficiency(value: float = 1.0) -> Callable[[float], float]:
    """Return a constant detection efficiency function."""

    def eff(_: float) -> float:
        return value

    return eff


def cebr3_efficiency(e_keV: np.ndarray | float) -> np.ndarray:
    """
    CeBr3-like detection efficiency model.

    - 0 below 30 keV
    - near 1.0 for 30-150 keV
    - decays with a power law above 150 keV
    """
    e = np.asarray(e_keV, dtype=float)
    eff = np.zeros_like(e, dtype=float)
    plateau = (e >= 30.0) & (e <= 150.0)
    eff[plateau] = 1.0
    high = e > 150.0
    eff[high] = (150.0 / np.maximum(e[high], 1e-9)) ** 0.6
    if eff.shape == ():
        return float(eff)
    return eff


def energy_dependent_efficiency(e_keV: np.ndarray | float) -> np.ndarray:
    """Backward-compatible alias."""
    return cebr3_efficiency(e_keV)


def build_response_matrix(
    energy_axis: NDArray[np.float64],
    library: Dict[str, Nuclide],
    resolution_fn: Callable[[float], float],
    efficiency_fn: Callable[[float], float],
    bin_width_keV: float | None = None,
    *,
    use_gpu: bool | None = None,
    gpu_device: str = "cuda",
    gpu_dtype: str = "float32",
) -> NDArray[np.float64]:
    """
    Build a response matrix from the nuclide library.

    Rows are energy bins, columns are nuclides, and entries represent expected counts
    per unit activity.

    GPU acceleration is enabled when use_gpu=True and torch supports the device.
    """
    if bin_width_keV is None:
        if energy_axis.size < 2:
            raise ValueError("energy_axis must contain at least two points to infer bin width")
        bin_width_keV = float(energy_axis[1] - energy_axis[0])
    num_bins = energy_axis.size
    num_iso = len(library)
    matrix = np.zeros((num_bins, num_iso), dtype=float)
    ctx = _resolve_torch_context(use_gpu, gpu_device, gpu_dtype)
    for col_idx, nuclide in enumerate(library.values()):
        if ctx is None:
            matrix[:, col_idx] = _nuclide_response(
                energy_axis, nuclide.lines, resolution_fn, efficiency_fn, bin_width_keV
            )
        else:
            matrix[:, col_idx] = _nuclide_response_torch(
                energy_axis,
                nuclide.lines,
                resolution_fn,
                efficiency_fn,
                bin_width_keV,
                ctx,
            )
    return matrix


def _nuclide_response(
    energy_axis: NDArray[np.float64],
    lines: Iterable[NuclideLine],
    resolution_fn: Callable[[float], float],
    efficiency_fn: Callable[[float], float],
    bin_width_keV: float,
) -> NDArray[np.float64]:
    """Compute the response for a single nuclide by summing its lines."""
    response = np.zeros_like(energy_axis, dtype=float)
    for line in lines:
        sigma = resolution_fn(line.energy_keV)
        peak = gaussian_peak(energy_axis, center=line.energy_keV, sigma=sigma)
        peak_area = peak.sum() * bin_width_keV
        # Add Compton continuum with the same area basis as the full-energy peak.
        cont_shape = compton_continuum_shape(energy_axis, line.energy_keV, shape="exponential")
        if cont_shape.sum() > 0:
            cont_shape = cont_shape / cont_shape.sum()
        cont = COMPTON_CONTINUUM_TO_PEAK * peak_area * cont_shape
        eff = efficiency_fn(line.energy_keV)
        peak *= eff
        cont *= eff
        response += line.intensity * peak * bin_width_keV
        response += line.intensity * cont
        # Add a backscatter peak for higher-energy lines.
        if line.energy_keV > 200.0:
            e_back = backscatter_energy(line.energy_keV)
            sigma_back = resolution_fn(e_back)
            back = gaussian_peak(energy_axis, center=e_back, sigma=sigma_back)
            back_norm = back.sum() * bin_width_keV
            if back_norm > 0:
                area_back = BACKSCATTER_FRACTION * peak_area
                back *= area_back / back_norm
                back *= efficiency_fn(e_back)
                response += line.intensity * back
    return response


def _nuclide_response_torch(
    energy_axis: NDArray[np.float64],
    lines: Iterable[NuclideLine],
    resolution_fn: Callable[[float], float],
    efficiency_fn: Callable[[float], float],
    bin_width_keV: float,
    ctx: tuple["torch.device", "torch.dtype"],
) -> NDArray[np.float64]:
    """Compute the response for a single nuclide using torch for vector ops."""
    if torch is None:
        raise RuntimeError("torch is not available")
    device, dtype = ctx
    energy_t = torch.as_tensor(energy_axis, device=device, dtype=dtype)
    response_t = torch.zeros_like(energy_t)
    for line in lines:
        sigma = float(resolution_fn(line.energy_keV))
        peak_t = _gaussian_peak_torch(energy_t, center=float(line.energy_keV), sigma=sigma)
        peak_area = torch.sum(peak_t) * float(bin_width_keV)
        cont_shape = _compton_continuum_shape_torch(energy_t, float(line.energy_keV), shape="exponential")
        cont_sum = torch.sum(cont_shape)
        if float(cont_sum) > 0.0:
            cont_shape = cont_shape / cont_sum
        cont_t = COMPTON_CONTINUUM_TO_PEAK * peak_area * cont_shape
        eff = float(efficiency_fn(line.energy_keV))
        peak_t = peak_t * eff
        cont_t = cont_t * eff
        response_t = response_t + line.intensity * peak_t * float(bin_width_keV)
        response_t = response_t + line.intensity * cont_t
        if line.energy_keV > 200.0:
            e_back = backscatter_energy(line.energy_keV)
            sigma_back = float(resolution_fn(e_back))
            back_t = _gaussian_peak_torch(energy_t, center=float(e_back), sigma=sigma_back)
            back_norm = torch.sum(back_t) * float(bin_width_keV)
            if float(back_norm) > 0.0:
                area_back = BACKSCATTER_FRACTION * peak_area
                back_t = back_t * (area_back / back_norm)
                back_t = back_t * float(efficiency_fn(e_back))
                response_t = response_t + line.intensity * back_t
    return response_t.detach().cpu().numpy()
