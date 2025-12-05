"""Real-time demo for the rotating-shield particle filter with visualization.

Run `python main.py` to simulate a robot moving through a simple environment, measuring
isotope-wise counts from synthetic spectra (Chapter 2.5.7) with rotating 1/8-shell
shields, updating the PF (Chapter 3), and visualizing the particle clouds and estimates.

Notes:
- PF observations are always the isotope-wise counts from spectrum unfolding (no direct
  geometric shortcuts).
- The PF algorithm itself is unchanged; this script focuses on live visualization and
  playback. A final snapshot is saved to results/result.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

# Prefer interactive backend; fallback to Agg when unavailable.
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Ensure src/ is on sys.path for direct script execution.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RESULTS_DIR = ROOT / "results"

from measurement.model import EnvironmentConfig, PointSource
from measurement.shielding import OctantShield, generate_octant_orientations, generate_octant_rotation_matrices
from spectrum.pipeline import SpectralDecomposer
from pf.parallel import ParallelIsotopePF, Measurement
from pf.particle_filter import PFConfig
from measurement.kernels import ShieldParams, KernelPrecomputer
from planning.shield_rotation import select_best_orientation
from visualization.realtime_viz import build_frame_from_pf, RealTimePFVisualizer


def _build_demo_sources() -> list[PointSource]:
    """Define a small set of synthetic sources inside the environment."""
    return [
        PointSource("Cs-137", position=(3.0, 3.0, 1.0), intensity_cps_1m=20000.0),
        PointSource("Co-60", position=(7.0, 4.0, 1.0), intensity_cps_1m=30000.0),
        PointSource("Eu-154", position=(6.0, 7.0, 1.0), intensity_cps_1m=40000.0),
    ]


def run_live_pf(live: bool = True, steps: int = 10, output_path: str = "result.png") -> None:
    """Run a simple PF loop with live visualization."""
    env = EnvironmentConfig(size_x=10.0, size_y=10.0, size_z=3.0, detector_position=(1.0, 1.0, 1.0))
    sources = _build_demo_sources()
    decomposer = SpectralDecomposer()
    octant_shield = OctantShield()
    normals = generate_octant_orientations()
    rot_mats = generate_octant_rotation_matrices()

    # Candidate sources: coarse grid inside environment
    xs = np.linspace(0.5, env.size_x - 0.5, 4)
    ys = np.linspace(0.5, env.size_y - 0.5, 4)
    zs = np.linspace(0.5, env.size_z - 0.5, 2)
    grid = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)

    # Define a simple trajectory (L-shaped)
    trajectory = [
        np.array([1.0, 1.0, 1.0]),
        np.array([3.0, 1.0, 1.0]),
        np.array([5.0, 1.0, 1.0]),
        np.array([5.0, 3.0, 1.0]),
        np.array([5.0, 5.0, 1.0]),
        np.array([7.0, 5.0, 1.0]),
        np.array([9.0, 5.0, 1.0]),
    ]

    pf_conf = PFConfig(num_particles=50, max_sources=2, resample_threshold=0.5, min_strength=0.01, p_birth=0.05)
    isotopes = ["Cs-137", "Co-60", "Eu-154"]
    pf = ParallelIsotopePF(isotope_names=isotopes, config=pf_conf)
    # Build kernel using all poses up-front
    poses_arr = np.array(trajectory, dtype=float)
    kernel = KernelPrecomputer(
        candidate_sources=grid,
        poses=poses_arr,
        orientations=normals,
        shield_params=ShieldParams(),
        mu_by_isotope={iso: 0.5 for iso in isotopes},
    )
    for iso in isotopes:
        pf.attach_kernel(iso, kernel)

    viz = RealTimePFVisualizer(isotopes=isotopes, world_bounds=(0, env.size_x, 0, env.size_y, 0, env.size_z))
    if live:
        plt.ion()

    latest_counts: dict[str, float] = {}
    elapsed = 0.0

    for k in range(min(steps, len(trajectory))):
        pose = trajectory[k]
        # Cycle orientations for visualization; replace with select_best_orientation if desired
        best_idx = k % len(normals)
        orient_vec = normals[best_idx]
        Rmat = rot_mats[best_idx]
        # Simulate measurement via spectrum unfolding
        env_step = EnvironmentConfig(detector_position=tuple(pose))
        spectrum, _ = decomposer.simulate_spectrum(
            sources=sources,
            environment=env_step,
            acquisition_time=0.5,
            rng=np.random.default_rng(123 + k),
            shield_orientation=orient_vec,
            octant_shield=octant_shield,
        )
        z_k = decomposer.isotope_counts(spectrum)
        latest_counts = z_k
        meas = Measurement(
            counts_by_isotope=z_k,
            pose_idx=k,
            orient_idx=best_idx,
            live_time_s=0.5,
            fe_index=best_idx,
            pb_index=best_idx,
            RFe=Rmat,
            RPb=Rmat,
            detector_position=pose,
        )
        pf.update_all(meas)
        elapsed += 0.5
        frame = build_frame_from_pf(pf, meas, step_index=k, time_sec=elapsed)
        viz.update(frame)
        # Log to console
        rfe_dir = Rmat[:, 2].tolist()
        print(f"[step {k}] pose={pose.tolist()} orient_idx={best_idx} RFe_dir={rfe_dir} counts={z_k}")
        if live:
            plt.pause(0.05)
        # Stop if simple convergence (particles stabilized) or reached steps
        if pf.has_converged(tau_conv=1e-3, window=3):
            print("Converged; stopping early.")
            break

    # Save final snapshot
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    viz.save_final(out_path.as_posix())
    print(f"Final visualization saved to: {out_path}")
    if live:
        plt.ioff()
        plt.show(block=True)
    plt.close("all")


def run_realtime_pf(scenario_path: str | None = None, output_path: str = "result.png") -> None:
    """
    Entry point for real-time PF + visualization.

    For now, scenario loading is simplified: if scenario_path is provided, it can be parsed
    in the future; currently the built-in demo trajectory/sources are used.
    """
    run_live_pf(live=True, steps=10, output_path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time rotating-shield PF visualization demo.")
    parser.add_argument("--steps", type=int, default=10, help="Number of trajectory steps to simulate.")
    parser.add_argument("--no-live", action="store_true", help="Disable interactive updating (still saves result.png).")
    parser.add_argument("--output", type=str, default="result.png", help="Path to save final snapshot.")
    parser.add_argument("--scenario", type=str, default=None, help="(Optional) scenario config path (JSON/YAML).")
    args = parser.parse_args()
    # For now scenario loading is a placeholder; default to built-in demo
    run_live_pf(live=not args.no_live, steps=args.steps, output_path=args.output)


if __name__ == "__main__":
    main()
