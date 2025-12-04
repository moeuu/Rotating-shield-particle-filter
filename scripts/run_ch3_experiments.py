"""Run Chapter 3-style Monte Carlo experiments for the rotating-shield PF.

This script assembles the main scenarios used in Chapter 3 (single/dual sources,
with and without shields), simulates Poisson counts via the kernel model, runs the
RotatingShieldPFEstimator, and logs pose-level errors and uncertainty metrics.

References:
- Measurement model and Poisson likelihood: Sec. 3.4.3 (Eq. for Λ_{k,h} and z_k).
- Shield rotation and information gain: Sec. 3.5.
- Convergence/uncertainty U: Sec. 3.6.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from measurement.kernels import KernelPrecomputer, ShieldParams
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator


@dataclass(frozen=True)
class TrueSource:
    """Ground-truth source defined on the candidate grid."""

    isotope: str
    candidate_index: int
    strength: float


@dataclass(frozen=True)
class ExperimentScenario:
    """Compact description of a Chapter-3 simulation scenario."""

    name: str
    description: str
    candidate_sources: np.ndarray
    poses: np.ndarray
    normals: np.ndarray
    mu_by_isotope: Dict[str, float]
    true_sources: List[TrueSource]
    background: Dict[str, float]
    live_time_s: float = 1.0
    num_particles: int = 200
    max_sources: int = 2
    resample_threshold: float = 0.5


def _build_estimator(scn: ExperimentScenario) -> RotatingShieldPFEstimator:
    isotopes = sorted(set(ts.isotope for ts in scn.true_sources))
    cfg = RotatingShieldPFConfig(
        num_particles=scn.num_particles,
        max_sources=scn.max_sources,
        resample_threshold=scn.resample_threshold,
    )
    est = RotatingShieldPFEstimator(
        isotopes=isotopes,
        candidate_sources=scn.candidate_sources,
        shield_normals=scn.normals,
        mu_by_isotope=scn.mu_by_isotope,
        pf_config=cfg,
        shield_params=ShieldParams(),
    )
    for pose in scn.poses:
        est.add_measurement_pose(pose)
    est._ensure_kernel_cache()
    return est


def _simulate_measurements(
    scn: ExperimentScenario, est: RotatingShieldPFEstimator, rng: np.random.Generator
) -> Iterable[Dict[str, float]]:
    """
    Simulate isotope-wise counts z_k using the same kernel and poses as the PF.

    Uses Λ_{k,h} from Sec. 3.4.3 with Poisson sampling.
    """
    kernel: KernelPrecomputer = est.kernel_cache  # type: ignore[assignment]
    source_strengths: Dict[str, np.ndarray] = {}
    for iso in est.isotopes:
        vec = np.zeros(scn.candidate_sources.shape[0], dtype=float)
        for ts in scn.true_sources:
            if ts.isotope == iso:
                vec[ts.candidate_index] += ts.strength
        source_strengths[iso] = vec

    for k, pose in enumerate(scn.poses):
        orient_idx = k % scn.normals.shape[0]
        z_k: Dict[str, float] = {}
        for iso in est.isotopes:
            lam = kernel.expected_counts(
                isotope=iso,
                pose_idx=k,
                orient_idx=orient_idx,
                source_strengths=source_strengths[iso],
                background=scn.background.get(iso, 0.0),
                live_time_s=scn.live_time_s,
            )
            z_k[iso] = float(rng.poisson(lam=lam))
        yield z_k


def _match_errors(
    est: RotatingShieldPFEstimator, scn: ExperimentScenario
) -> tuple[float, float, float]:
    """Compute mean position/strength error and isotopic accuracy."""
    estimates = est.estimates()
    pos_errs = []
    str_errs = []
    iso_hits = 0
    for ts in scn.true_sources:
        cand_pos = scn.candidate_sources[ts.candidate_index]
        if ts.isotope not in estimates:
            pos_errs.append(np.inf)
            str_errs.append(np.inf)
            continue
        pos_est, str_est = estimates[ts.isotope]
        if pos_est.size == 0:
            pos_errs.append(np.inf)
            str_errs.append(np.inf)
            continue
        # nearest estimated source on the grid
        dists = np.linalg.norm(pos_est - cand_pos, axis=1)
        idx = int(np.argmin(dists))
        pos_errs.append(float(dists[idx]))
        str_errs.append(float(abs(str_est[idx] - ts.strength)))
        iso_hits += 1
    mean_pos = float(np.mean(pos_errs)) if pos_errs else 0.0
    mean_str = float(np.mean(str_errs)) if str_errs else 0.0
    iso_acc = iso_hits / max(len(scn.true_sources), 1)
    return mean_pos, mean_str, iso_acc


def run_single_trial(scn: ExperimentScenario, seed: int) -> Dict[str, float]:
    """Run one Monte Carlo trial and return logged metrics."""
    rng = np.random.default_rng(seed)
    est = _build_estimator(scn)
    for k, z_k in enumerate(_simulate_measurements(scn, est, rng)):
        est.update(z_k=z_k, pose_idx=k, orient_idx=k % scn.normals.shape[0], live_time_s=scn.live_time_s)
    pos_err, str_err, iso_acc = _match_errors(est, scn)
    return {
        "scenario": scn.name,
        "seed": seed,
        "position_error": pos_err,
        "strength_error": str_err,
        "iso_accuracy": iso_acc,
        "num_measurements": len(scn.poses),
        "num_orientations": scn.normals.shape[0],
        "global_uncertainty": est.global_uncertainty(),
    }


def default_scenarios() -> List[ExperimentScenario]:
    """Representative scenarios mirroring Chapter 3 experiments."""
    candidate_sources = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    poses = np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=float)
    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)
    mu = {"Cs-137": 0.5, "Co-60": 0.6}
    return [
        ExperimentScenario(
            name="single_cs_shielded",
            description="Single Cs-137 with rotating shield (Sec. 3.5 style).",
            candidate_sources=candidate_sources,
            poses=poses,
            normals=normals,
            mu_by_isotope=mu,
            true_sources=[TrueSource("Cs-137", candidate_index=1, strength=20.0)],
            background={"Cs-137": 0.1},
            live_time_s=1.0,
            num_particles=150,
            max_sources=1,
        ),
        ExperimentScenario(
            name="dual_iso_shielded",
            description="Cs-137 + Co-60 at different grid points.",
            candidate_sources=candidate_sources,
            poses=poses,
            normals=normals,
            mu_by_isotope=mu,
            true_sources=[
                TrueSource("Cs-137", candidate_index=0, strength=15.0),
                TrueSource("Co-60", candidate_index=2, strength=12.0),
            ],
            background={"Cs-137": 0.1, "Co-60": 0.1},
            live_time_s=1.0,
            num_particles=200,
            max_sources=2,
        ),
    ]


def run_experiments(output_dir: Path, seeds: Iterable[int] | None = None, scenarios: Iterable[ExperimentScenario] | None = None) -> None:
    """Run all scenarios for given seeds and append JSONL logs to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds) if seeds is not None else list(range(5))
    scenarios = list(scenarios) if scenarios is not None else default_scenarios()
    for scn in scenarios:
        log_path = output_dir / f"{scn.name}.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            for seed in seeds:
                metrics = run_single_trial(scn, seed)
                f.write(json.dumps(metrics) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 3 PF experiments.")
    parser.add_argument("--output", type=Path, default=Path("results/ch3_experiments"), help="Directory to store JSONL logs.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4], help="Random seeds to run.")
    args = parser.parse_args()
    run_experiments(output_dir=args.output, seeds=args.seeds)
    print(f"Logs written to: {args.output}")


if __name__ == "__main__":
    main()
