"""Choose the next robot pose while balancing uncertainty reduction and motion cost (Sec. 3.5.4)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import sys
import threading
import time

import numpy as np
from numpy.typing import NDArray

from pf.estimator import RotatingShieldPFEstimator
from runtime_defaults import DEFAULT_MEASUREMENT_TIME_S


DEFAULT_PLANNING_ROLLOUTS = 8


@contextmanager
def _temporary_gpu_settings(
    estimator: RotatingShieldPFEstimator,
    use_gpu: bool | None,
    gpu_device: str | None,
    gpu_dtype: str | None,
) -> None:
    """Temporarily override estimator GPU settings for planning evaluations."""
    if use_gpu is None and gpu_device is None and gpu_dtype is None:
        yield
        return
    pf_config = getattr(estimator, "pf_config", None)
    if pf_config is None:
        yield
        return
    prior_use_gpu = pf_config.use_gpu
    prior_device = pf_config.gpu_device
    prior_dtype = pf_config.gpu_dtype
    if use_gpu is not None:
        pf_config.use_gpu = bool(use_gpu)
    if gpu_device is not None:
        pf_config.gpu_device = str(gpu_device)
    if gpu_dtype is not None:
        pf_config.gpu_dtype = str(gpu_dtype)
    try:
        yield
    finally:
        pf_config.use_gpu = prior_use_gpu
        pf_config.gpu_device = prior_device
        pf_config.gpu_dtype = prior_dtype


def estimate_lambda_cost(
    uncertainties: NDArray[np.float64],
    motion_costs: NDArray[np.float64],
    *,
    method: str = "range",
    scale: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """
    Estimate lambda_cost by matching uncertainty and motion-cost scales.

    method:
        - "range": use max-min
        - "iqr": use interquartile range
    """
    uncertainties = np.asarray(uncertainties, dtype=float).ravel()
    motion_costs = np.asarray(motion_costs, dtype=float).ravel()
    if uncertainties.size == 0 or motion_costs.size == 0:
        return 0.0
    if method == "range":
        u_scale = float(np.ptp(uncertainties))
        d_scale = float(np.ptp(motion_costs))
    elif method == "iqr":
        u_scale = float(
            np.quantile(uncertainties, 0.75) - np.quantile(uncertainties, 0.25)
        )
        d_scale = float(
            np.quantile(motion_costs, 0.75) - np.quantile(motion_costs, 0.25)
        )
    else:
        raise ValueError(f"Unknown lambda_cost method: {method}")
    if u_scale <= eps or d_scale <= eps:
        return 0.0
    return float(scale) * (u_scale / d_scale)


def minimum_observation_shortfall(
    counts_by_isotope: dict[str, float],
    min_counts: float,
) -> float:
    """
    Return a dimensionless soft-constraint penalty for isotope observability.

    The penalty is zero only when every candidate isotope has at least
    ``min_counts`` expected counts. It is the mean squared relative shortfall,
    so it is independent of the absolute count scale.
    """
    min_counts = float(min_counts)
    if min_counts <= 0.0:
        return 0.0
    if not counts_by_isotope:
        return 1.0
    shortfalls = [
        max(0.0, 1.0 - max(float(count), 0.0) / min_counts) ** 2
        for count in counts_by_isotope.values()
    ]
    return float(np.mean(shortfalls)) if shortfalls else 0.0


def _minimum_observation_feasible_mask(
    penalties: NDArray[np.float64],
    min_counts: float,
    eps: float = 1e-12,
) -> NDArray[np.bool_]:
    """Return candidates satisfying the all-isotope minimum observation target."""
    penalty_arr = np.asarray(penalties, dtype=float).ravel()
    if float(min_counts) <= 0.0:
        return np.ones(penalty_arr.shape, dtype=bool)
    return penalty_arr <= float(eps)


def _auto_scale_observation_penalty(
    base_scores: NDArray[np.float64],
    penalties: NDArray[np.float64],
    scale: float,
    eps: float = 1e-12,
) -> float:
    """Return a score-compatible weight for dimensionless observation penalties."""
    scale = max(float(scale), 0.0)
    if scale <= 0.0 or penalties.size == 0 or float(np.max(penalties)) <= eps:
        return 0.0
    base_scores = np.asarray(base_scores, dtype=float)
    penalties = np.asarray(penalties, dtype=float)
    base_range = float(np.ptp(base_scores))
    if base_range <= eps:
        base_range = max(float(np.mean(np.abs(base_scores))), 1.0)
    penalty_range = float(np.ptp(penalties))
    if penalty_range <= eps:
        return scale * base_range
    return scale * base_range / penalty_range


def select_next_pose_from_candidates(
    estimator: RotatingShieldPFEstimator,
    candidate_poses_xyz: NDArray[np.float64],
    current_pose_xyz: NDArray[np.float64],
    *,
    lambda_cost: float | None = None,
    tau_ig: float | None = None,
    t_max_s: float | None = None,
    t_short_s: float = DEFAULT_MEASUREMENT_TIME_S,
    num_rollouts: int = 0,
    use_mean_measurement: bool = True,
    rng_seed: int | None = 0,
    verbose: bool = False,
    progress_every: int = 50,
    preview_k: int = 5,
    use_gpu: bool | None = None,
    gpu_device: str | None = None,
    gpu_dtype: str | None = None,
    auto_lambda_cost: bool = False,
    lambda_cost_method: str = "range",
    lambda_cost_scale: float = 1.0,
    top_k: int = 5,
    ig_breakdown_k: int | None = None,
    ig_breakdown_max_steps: int = 6,
    ig_breakdown_max_rollouts: int = 2,
    min_observation_counts: float = 0.0,
    min_observation_penalty_scale: float = 1.0,
    min_observation_aggregate: str = "max",
    min_observation_max_particles: int | None = None,
    worker_count: int | None = None,
) -> int:
    """
    Select the next pose from explicit candidates using after-rotation uncertainty.

    Score_k = E[U_after-rotation | q_k] + lambda_cost * C_move + eta * G(q_k).
    If any candidate satisfies the all-isotope ``min_observation_counts``
    target, infeasible candidates are excluded. If none do, G(q_k) remains a
    soft fallback penalty so planning does not deadlock.

    GPU settings can be overridden for planning with use_gpu/gpu_device/gpu_dtype.
    When verbose is True, top_k and ig_breakdown_k control extra diagnostics.
    If ig_breakdown_k is None, the IG breakdown is reported for top_k candidates.
    If auto_lambda_cost is True, lambda_cost is computed from candidate scales.
    """

    def _evaluate_candidate(
        item: tuple[int, NDArray[np.float64]],
    ) -> tuple[int, float, float, dict[str, float], float]:
        """Return uncertainty, motion cost, and observation penalty for one pose."""
        idx, pose_eval = item
        uncertainty_eval = estimator.expected_uncertainty_after_rotation(
            pose_xyz=pose_eval,
            live_time_per_rot_s=t_short_s,
            tau_ig=tau_ig,
            tmax_s=t_max_s,
            n_rollouts=rollouts,
            orient_selection="IG",
            rng_seed=int(candidate_seeds[idx]),
        )
        motion_cost_eval = float(np.linalg.norm(pose_eval - current_pose_xyz))
        if min_observation_counts > 0.0:
            counts_by_iso_eval = (
                estimator.expected_observation_counts_by_isotope_at_pose(
                    pose_eval,
                    live_time_s=t_short_s,
                    aggregate=min_observation_aggregate,
                    max_particles=observation_max_particles,
                )
            )
            observation_penalty_eval = minimum_observation_shortfall(
                counts_by_iso_eval,
                min_counts=float(min_observation_counts),
            )
        else:
            counts_by_iso_eval = {}
            observation_penalty_eval = 0.0
        return (
            int(idx),
            float(uncertainty_eval),
            float(motion_cost_eval),
            counts_by_iso_eval,
            float(observation_penalty_eval),
        )

    def _spinner_worker(
        stop_event: threading.Event,
        base_label: str,
        start_time: float,
        width: int,
    ) -> None:
        """Render a spinner with a timer until stop_event is set."""
        frame_idx = 0
        while not stop_event.is_set():
            frame = spinner[frame_idx % len(spinner)]
            frame_idx += 1
            elapsed = time.monotonic() - start_time
            label = f"{base_label} t={elapsed:7.1f}s"
            if len(label) < width:
                label = label + " " * (width - len(label))
            sys.stdout.write(f"\r{frame} {label}")
            sys.stdout.flush()
            stop_event.wait(0.1)

    with _temporary_gpu_settings(estimator, use_gpu, gpu_device, gpu_dtype):
        candidate_poses_xyz = np.asarray(candidate_poses_xyz, dtype=float)
        if candidate_poses_xyz.ndim != 2 or candidate_poses_xyz.shape[1] != 3:
            raise ValueError("candidate_poses_xyz must be shape (N, 3).")
        if candidate_poses_xyz.shape[0] == 0:
            raise ValueError("candidate_poses_xyz must contain at least one pose.")
        current_pose_xyz = np.asarray(current_pose_xyz, dtype=float)
        if verbose:
            total = int(candidate_poses_xyz.shape[0])
            preview = candidate_poses_xyz[: min(int(preview_k), total)]
            preview_str = np.array2string(preview, precision=3, separator=", ")
            print(f"Selecting next pose from {total} candidates.")
            if preview.size:
                print(f"Candidate preview (first {len(preview)}): {preview_str}")
        pf_config = getattr(estimator, "pf_config", None)
        if lambda_cost is None:
            lam_cost = pf_config.lambda_cost if pf_config is not None else 1.0
        else:
            lam_cost = float(lambda_cost)
        tau_ig = (
            pf_config.ig_threshold if pf_config is not None else 1e-3
        ) if tau_ig is None else tau_ig
        t_max_s = (
            pf_config.max_dwell_time_s if pf_config is not None else 1.0
        ) if t_max_s is None else t_max_s
        t_short_s = float(t_short_s)
        if not np.isfinite(t_short_s) or t_short_s <= 0.0:
            raise ValueError("t_short_s must be finite and positive.")
        seed_rng = (
            np.random.default_rng(rng_seed)
            if rng_seed is not None
            else np.random.default_rng()
        )
        rollouts = int(num_rollouts)
        if rollouts <= 0 and not use_mean_measurement:
            rollouts = 1
        observation_max_particles = min_observation_max_particles
        if observation_max_particles is None and pf_config is not None:
            planning_limit = getattr(pf_config, "planning_rollout_particles", None)
            if planning_limit is None:
                planning_limit = getattr(pf_config, "planning_particles", None)
            if planning_limit is not None:
                observation_max_particles = max(1, int(planning_limit))
        elif observation_max_particles is not None:
            observation_max_particles = max(1, int(observation_max_particles))
        uncertainties = []
        motion_costs = []
        observation_penalties = []
        observation_counts_by_candidate: list[dict[str, float]] = []
        spinner = ["|", "/", "-", "\\"]
        last_line_len = 0
        total_candidates = int(len(candidate_poses_xyz))
        candidate_seeds = seed_rng.integers(
            0,
            2**32 - 1,
            size=total_candidates,
            dtype=np.uint32,
        )
        gpu_enabled = bool(use_gpu) if use_gpu is not None else False
        if pf_config is not None:
            gpu_enabled = bool(getattr(pf_config, "use_gpu", gpu_enabled))
        if worker_count is None:
            configured_workers = getattr(pf_config, "pose_selection_workers", None)
            if configured_workers is None:
                configured_workers = getattr(pf_config, "ig_workers", 1)
        else:
            configured_workers = worker_count
        try:
            candidate_workers = max(1, int(configured_workers))
        except (TypeError, ValueError):
            candidate_workers = 1
        if gpu_enabled or total_candidates <= 1:
            candidate_workers = 1
        if verbose and total_candidates > 1:
            worker_note = (
                "serial" if candidate_workers == 1 else f"{candidate_workers} workers"
            )
            print(f"Candidate evaluation mode: {worker_note}.")
        if candidate_workers > 1:
            indexed_candidates = [
                (idx, np.asarray(pose, dtype=float).copy())
                for idx, pose in enumerate(candidate_poses_xyz)
            ]
            with ThreadPoolExecutor(max_workers=candidate_workers) as executor:
                results = list(executor.map(_evaluate_candidate, indexed_candidates))
            results.sort(key=lambda item: int(item[0]))
            for _, uncertainty, motion_cost, counts_by_iso, obs_penalty in results:
                uncertainties.append(float(uncertainty))
                motion_costs.append(float(motion_cost))
                observation_counts_by_candidate.append(dict(counts_by_iso))
                observation_penalties.append(float(obs_penalty))
        else:
            for idx, pose in enumerate(candidate_poses_xyz):
                should_report = (
                    verbose
                    and progress_every > 0
                    and (
                        (idx + 1) % progress_every == 0
                        or (idx + 1) == total_candidates
                    )
                )
                stop_event = None
                spinner_thread = None
                if should_report:
                    pose_preview = np.array2string(pose, precision=3, separator=", ")
                    base_label = (
                        f"evaluating candidate {idx + 1}/{total_candidates} "
                        f"pose={pose_preview}"
                    )
                    label_width = len(base_label) + len(" t=0000.0s")
                    last_line_len = max(last_line_len, label_width)
                    stop_event = threading.Event()
                    start_time = time.monotonic()
                    spinner_thread = threading.Thread(
                        target=_spinner_worker,
                        args=(stop_event, base_label, start_time, last_line_len),
                        daemon=True,
                    )
                    spinner_thread.start()
                (
                    _idx,
                    uncertainty,
                    motion_cost,
                    counts_by_iso,
                    obs_penalty,
                ) = _evaluate_candidate((idx, pose))
                if spinner_thread is not None and stop_event is not None:
                    stop_event.set()
                    spinner_thread.join()
                uncertainties.append(float(uncertainty))
                motion_costs.append(float(motion_cost))
                observation_counts_by_candidate.append(dict(counts_by_iso))
                observation_penalties.append(float(obs_penalty))
        if verbose and progress_every > 0 and len(candidate_poses_xyz) > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
        uncertainties_arr = np.asarray(uncertainties, dtype=float)
        motion_costs_arr = np.asarray(motion_costs, dtype=float)
        if auto_lambda_cost:
            lam_cost = estimate_lambda_cost(
                uncertainties_arr,
                motion_costs_arr,
                method=lambda_cost_method,
                scale=lambda_cost_scale,
            )
        base_scores = uncertainties_arr + lam_cost * motion_costs_arr
        observation_penalties_arr = np.asarray(observation_penalties, dtype=float)
        observation_penalty_weight = _auto_scale_observation_penalty(
            base_scores,
            observation_penalties_arr,
            scale=float(min_observation_penalty_scale),
        )
        scores = base_scores + observation_penalty_weight * observation_penalties_arr
        feasible_mask = _minimum_observation_feasible_mask(
            observation_penalties_arr,
            float(min_observation_counts),
        )
        feasible_count = int(np.count_nonzero(feasible_mask))
        if 0 < feasible_count < int(scores.size):
            scores = scores.copy()
            scores[~feasible_mask] = np.inf
        best_idx = int(np.argmin(scores))
        if verbose and best_idx >= 0:
            best_pose = candidate_poses_xyz[best_idx]
            if auto_lambda_cost:
                print(
                    "Auto lambda_cost: "
                    f"value={lam_cost:.6g} method={lambda_cost_method} "
                    f"scale={lambda_cost_scale:.6g}"
                )
            print(
                "Best candidate selected: "
                f"idx={best_idx}, pose={best_pose.tolist()}, "
                f"uncertainty={uncertainties_arr[best_idx]:.6g}, "
                f"motion_cost={motion_costs_arr[best_idx]:.6g}, "
                f"observation_penalty={observation_penalties_arr[best_idx]:.6g}, "
                f"score={scores[best_idx]:.6g}"
            )
            print(
                "Selection reason: "
                f"minimum score among {len(scores)} candidates "
                f"(score = uncertainty + {lam_cost:.6g} * motion_cost "
                f"+ {observation_penalty_weight:.6g} * observation_penalty)."
            )
            if min_observation_counts > 0.0:
                print(
                    "Observation guarantee: "
                    f"min_counts={float(min_observation_counts):.6g}, "
                    f"aggregate={min_observation_aggregate}, "
                    f"feasible_candidates={feasible_count}/{len(scores)}, "
                    f"counts={observation_counts_by_candidate[best_idx]}"
                )
            if len(scores) > 1:
                order = np.argsort(scores)
                runner_up_idx = int(order[1])
                delta = float(scores[runner_up_idx] - scores[best_idx])
                runner_pose = candidate_poses_xyz[runner_up_idx]
                print(
                    "Runner-up: "
                    f"idx={runner_up_idx}, pose={runner_pose.tolist()}, "
                    f"uncertainty={uncertainties_arr[runner_up_idx]:.6g}, "
                    f"motion_cost={motion_costs_arr[runner_up_idx]:.6g}, "
                    f"observation_penalty={observation_penalties_arr[runner_up_idx]:.6g}, "
                    f"score={scores[runner_up_idx]:.6g}, Δscore={delta:.6g}"
                )
        if verbose and top_k > 0 and scores.size:
            order = np.argsort(scores)
            top_k = min(int(top_k), len(order))
            print(f"Top {top_k} candidates by score:")
            for rank, idx in enumerate(order[:top_k], start=1):
                print(
                    f"  #{rank} idx={int(idx)} pose={candidate_poses_xyz[int(idx)].tolist()} "
                    f"uncertainty={uncertainties[int(idx)]:.6g} "
                    f"motion_cost={motion_costs[int(idx)]:.6g} "
                    f"observation_penalty={observation_penalties_arr[int(idx)]:.6g} "
                    f"score={scores[int(idx)]:.6g}"
                )
        if verbose and ig_breakdown_k is None:
            ig_breakdown_k = top_k
        if (
            verbose
            and ig_breakdown_k is not None
            and ig_breakdown_k > 0
            and scores.size
        ):
            order = np.argsort(scores)
            ig_breakdown_k = min(int(ig_breakdown_k), len(order))
            ig_breakdown_max_steps = max(int(ig_breakdown_max_steps), 1)
            ig_breakdown_max_rollouts = max(int(ig_breakdown_max_rollouts), 1)
            print(f"IG breakdown for top {ig_breakdown_k} candidates:")
            for rank, idx in enumerate(order[:ig_breakdown_k], start=1):
                seed = int(candidate_seeds[int(idx)])
                u_val, debug = estimator.expected_uncertainty_after_rotation(
                    pose_xyz=candidate_poses_xyz[int(idx)],
                    live_time_per_rot_s=t_short_s,
                    tau_ig=tau_ig,
                    tmax_s=t_max_s,
                    n_rollouts=rollouts,
                    orient_selection="IG",
                    return_debug=True,
                    rng_seed=seed,
                )
                rollouts_debug = debug.get("rollouts", [])
                print(
                    f"  #{rank} idx={int(idx)} pose={candidate_poses_xyz[int(idx)].tolist()} "
                    f"uncertainty={u_val:.6g}"
                )
                if not rollouts_debug:
                    print("    no rollout IG data")
                    continue
                for r_idx, rollout in enumerate(
                    rollouts_debug[:ig_breakdown_max_rollouts], start=1
                ):
                    ig_vals = [step["ig"] for step in rollout.get("iterations", [])]
                    if not ig_vals:
                        print(f"    rollout {r_idx}: no IG steps")
                        continue
                    ig_trim = ig_vals[:ig_breakdown_max_steps]
                    ig_str = ", ".join(f"{val:.4g}" for val in ig_trim)
                    suffix = " ..." if len(ig_vals) > ig_breakdown_max_steps else ""
                    mean_ig = float(np.mean(ig_vals))
                    print(f"    rollout {r_idx}: ig=[{ig_str}]{suffix} mean={mean_ig:.4g}")
        return int(np.argmin(scores))
