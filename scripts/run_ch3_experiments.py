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

from evaluation_metrics import POSITION_ERROR_TARGET_M, compute_metrics
from measurement.model import EnvironmentConfig, PointSource
from measurement.kernels import ShieldParams
from pf.estimator import RotatingShieldPFConfig, RotatingShieldPFEstimator
from spectrum.pipeline import SpectralDecomposer


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
    observation_mu_by_isotope: Dict[str, float] | None = None
    mismatch_label: str = "matched"
    match_radius_m: float = POSITION_ERROR_TARGET_M
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
) -> Iterable[tuple[Dict[str, float], int, int]]:
    """
    Simulate isotope-wise counts z_k using the same kernel and poses as the PF.

    Uses spectral simulation + unfolding (Sec. 2.5.7 + Sec. 3.4.3) so PF observes
    isotope-wise counts derived from spectra, not direct kernel means.
    """
    decomposer = SpectralDecomposer()

    for k, pose in enumerate(scn.poses):
        num_orient = scn.normals.shape[0]
        fe_index = k % num_orient
        pb_index = (k // num_orient) % num_orient
        env = EnvironmentConfig(detector_position=tuple(pose.tolist()))
        fe_orientation = scn.normals[fe_index]
        pb_orientation = scn.normals[pb_index]
        sources = [
            PointSource(
                ts.isotope,
                position=tuple(scn.candidate_sources[ts.candidate_index].tolist()),
                intensity_cps_1m=ts.strength,
            )
            for ts in scn.true_sources
        ]
        spectrum, _ = decomposer.simulate_spectrum(
            sources=sources,
            environment=env,
            acquisition_time=scn.live_time_s,
            rng=rng,
            fe_shield_orientation=fe_orientation,
            pb_shield_orientation=pb_orientation,
            mu_by_isotope=scn.observation_mu_by_isotope or scn.mu_by_isotope,
            shield_params=ShieldParams(),
        )
        z_k = decomposer.isotope_counts(spectrum)
        yield z_k, fe_index, pb_index


def _serialize_mu_map(mu_by_isotope: Dict[str, float] | None) -> Dict[str, float]:
    """Return a JSON-serializable attenuation map."""
    if mu_by_isotope is None:
        return {}
    return {iso: float(value) for iso, value in mu_by_isotope.items()}


def _true_sources_by_isotope(scn: ExperimentScenario) -> Dict[str, List[Dict[str, object]]]:
    """Return ground-truth sources grouped by isotope for evaluation."""
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for src in scn.true_sources:
        position = scn.candidate_sources[src.candidate_index]
        grouped.setdefault(src.isotope, []).append(
            {
                "pos": [float(coord) for coord in position.tolist()],
                "strength": float(src.strength),
            }
        )
    return grouped


def _estimated_sources_by_isotope(est: RotatingShieldPFEstimator) -> Dict[str, List[Dict[str, object]]]:
    """Return estimator outputs grouped by isotope for evaluation."""
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for iso, (positions, strengths) in est.estimates().items():
        grouped[iso] = [
            {
                "pos": [float(coord) for coord in pos.tolist()],
                "strength": float(strength),
            }
            for pos, strength in zip(
                np.asarray(positions, dtype=float),
                np.asarray(strengths, dtype=float),
            )
        ]
    return grouped


def _summary_stats(values: List[float]) -> Dict[str, float | None]:
    """Return mean/median/RMSE/max statistics for a scalar list."""
    if not values:
        return {
            "mean": None,
            "median": None,
            "rmse": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "rmse": float(np.sqrt(np.mean(arr**2))),
        "max": float(np.max(arr)),
    }


def _summarize_trial_metrics(
    est: RotatingShieldPFEstimator,
    scn: ExperimentScenario,
) -> Dict[str, object]:
    """Compute aggregate trial metrics from the shared evaluator output."""
    gt_by_iso = _true_sources_by_isotope(scn)
    est_by_iso = _estimated_sources_by_isotope(est)
    metrics = compute_metrics(gt_by_iso, est_by_iso, match_radius_m=scn.match_radius_m)

    position_errors: List[float] = []
    strength_errors: List[float] = []
    gt_source_count = 0
    estimated_source_count = 0
    assigned_count = 0
    within_radius_match_count = 0
    isotope_metrics: Dict[str, Dict[str, object]] = {}

    for iso, iso_metrics in metrics["isotopes"].items():
        counts = iso_metrics["counts"]
        gt_count = int(counts["gt"])
        est_count = int(counts["est"])
        assigned = int(counts["assigned"])
        gt_source_count += gt_count
        estimated_source_count += est_count
        assigned_count += assigned
        within_radius = sum(1 for match in iso_metrics["matches"] if bool(match["within_radius"]))
        within_radius_match_count += within_radius

        for match in iso_metrics["matches"]:
            position_errors.append(float(match["distance"]))
            strength_errors.append(float(match["abs_err"]))

        isotope_metrics[iso] = {
            "gt_count": gt_count,
            "estimated_count": est_count,
            "assigned_count": assigned,
            "within_radius_match_count": within_radius,
            "position_error_mean": iso_metrics["position_error"]["mean"],
            "position_within_target": iso_metrics["position_error"]["within_target"],
        }

    position_summary = _summary_stats(position_errors)
    strength_summary = _summary_stats(strength_errors)
    fp_count = max(0, estimated_source_count - within_radius_match_count)
    fn_count = max(0, gt_source_count - within_radius_match_count)
    position_target_m = float(POSITION_ERROR_TARGET_M)
    mean_position_error = position_summary["mean"]
    position_within_target = bool(
        mean_position_error is not None
        and mean_position_error <= position_target_m
        and fn_count == 0
    )
    source_count_exact = bool(fp_count == 0 and fn_count == 0)
    trial_success = bool(position_within_target and source_count_exact)
    iso_accuracy = float(within_radius_match_count / gt_source_count) if gt_source_count else 1.0

    return {
        "position_error": position_summary["mean"],
        "position_error_median": position_summary["median"],
        "position_error_rmse": position_summary["rmse"],
        "position_error_max": position_summary["max"],
        "position_target_m": position_target_m,
        "position_within_target": position_within_target,
        "strength_error": strength_summary["mean"],
        "strength_error_median": strength_summary["median"],
        "strength_error_rmse": strength_summary["rmse"],
        "strength_error_max": strength_summary["max"],
        "iso_accuracy": iso_accuracy,
        "gt_source_count": gt_source_count,
        "estimated_source_count": estimated_source_count,
        "assigned_count": assigned_count,
        "within_radius_match_count": within_radius_match_count,
        "fp_count": fp_count,
        "fn_count": fn_count,
        "source_count_exact": source_count_exact,
        "trial_success": trial_success,
        "isotope_metrics": isotope_metrics,
    }


def run_single_trial(scn: ExperimentScenario, seed: int) -> Dict[str, object]:
    """Run one Monte Carlo trial and return logged metrics."""
    rng = np.random.default_rng(seed)
    est = _build_estimator(scn)
    for k, (z_k, fe_index, pb_index) in enumerate(_simulate_measurements(scn, est, rng)):
        est.update_pair(
            z_k=z_k,
            pose_idx=k,
            fe_index=fe_index,
            pb_index=pb_index,
            live_time_s=scn.live_time_s,
        )
    metrics = _summarize_trial_metrics(est, scn)
    return {
        "scenario": scn.name,
        "seed": seed,
        "mismatch_label": scn.mismatch_label,
        "num_measurements": len(scn.poses),
        "num_orientations": scn.normals.shape[0],
        "global_uncertainty": float(est.global_uncertainty()),
        "estimator_mu_by_isotope": _serialize_mu_map(scn.mu_by_isotope),
        "observation_mu_by_isotope": _serialize_mu_map(scn.observation_mu_by_isotope or scn.mu_by_isotope),
        **metrics,
    }


def _scaled_mu_map(mu_by_isotope: Dict[str, float], scale: float) -> Dict[str, float]:
    """Return an attenuation map scaled by a constant factor."""
    return {iso: float(value) * float(scale) for iso, value in mu_by_isotope.items()}


def default_scenarios() -> List[ExperimentScenario]:
    """Representative scenarios mirroring Chapter 3 experiments."""
    candidate_sources = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    poses = np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=float)
    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)
    mu = {"Cs-137": 0.5, "Co-60": 0.6}
    scenarios = [
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
    return scenarios + [
        ExperimentScenario(
            name=f"{scenario.name}_obs_mu_plus25",
            description=f"{scenario.description} Observation attenuation scaled by +25%.",
            candidate_sources=scenario.candidate_sources,
            poses=scenario.poses,
            normals=scenario.normals,
            mu_by_isotope=scenario.mu_by_isotope,
            observation_mu_by_isotope=_scaled_mu_map(scenario.mu_by_isotope, 1.25),
            true_sources=scenario.true_sources,
            background=scenario.background,
            mismatch_label="observation_mu_plus25",
            match_radius_m=scenario.match_radius_m,
            live_time_s=scenario.live_time_s,
            num_particles=scenario.num_particles,
            max_sources=scenario.max_sources,
            resample_threshold=scenario.resample_threshold,
        )
        for scenario in scenarios
    ]


def run_experiments(
    output_dir: Path,
    seeds: Iterable[int] | None = None,
    scenarios: Iterable[ExperimentScenario] | None = None,
) -> None:
    """Run all scenarios for given seeds and append JSONL logs to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds) if seeds is not None else list(range(5))
    scenarios = list(scenarios) if scenarios is not None else default_scenarios()
    for scn in scenarios:
        log_path = output_dir / f"{scn.name}.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            for seed in seeds:
                metrics = run_single_trial(scn, seed)
                f.write(json.dumps(metrics, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 3 PF experiments.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ch3_experiments"),
        help="Directory to store JSONL logs.",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4], help="Random seeds to run.")
    args = parser.parse_args()
    run_experiments(output_dir=args.output, seeds=args.seeds)
    print(f"Logs written to: {args.output}")


if __name__ == "__main__":
    main()
