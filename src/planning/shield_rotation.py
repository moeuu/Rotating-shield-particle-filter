"""Shield orientation selection based on information metrics (Sec. 3.4)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from measurement.continuous_kernels import ContinuousKernel
from pf.estimator import RotatingShieldPFEstimator


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Return a normalized copy of weights."""
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.ones_like(weights) / max(len(weights), 1)
    return weights / total


def _surrogate_scores(
    estimator: RotatingShieldPFEstimator,
    pose_idx: int,
    live_time_s: float,
    particles_by_isotope: Dict[str, Tuple[list, np.ndarray]],
    RFe_candidates: np.ndarray,
    RPb_candidates: np.ndarray,
    alpha_by_isotope: Dict[str, float] | None,
    allowed_indices: set[int] | None,
    metric: str,
) -> Dict[int, float]:
    """
    Compute cheap surrogate scores for all Fe/Pb orientation pairs.

    metric:
        - "var_lambda": weighted variance of Λ
        - "var_log_lambda": weighted variance of log(Λ+eps)
    """
    if not particles_by_isotope:
        return {}
    kernel = ContinuousKernel(mu_by_isotope=estimator.mu_by_isotope, shield_params=estimator.shield_params)
    alphas = alpha_by_isotope or {iso: 1.0 for iso in particles_by_isotope}
    alpha_sum = sum(alphas.values()) or 1.0
    alphas = {k: v / alpha_sum for k, v in alphas.items()}
    eps = 1e-12
    scores: Dict[int, float] = {}
    for fe_idx, RFe in enumerate(RFe_candidates):
        for pb_idx, RPb in enumerate(RPb_candidates):
            oid = fe_idx * len(RPb_candidates) + pb_idx
            if allowed_indices is not None and oid not in allowed_indices:
                continue
            score = 0.0
            for iso, (states, weights) in particles_by_isotope.items():
                weights = _normalize_weights(np.asarray(weights, dtype=float))
                lam = np.zeros(len(states), dtype=float)
                for i, st in enumerate(states):
                    lam[i] = kernel.expected_counts_pair(
                        isotope=iso,
                        detector_pos=estimator.poses[pose_idx],
                        sources=st.positions,
                        strengths=st.strengths,
                        fe_index=fe_idx,
                        pb_index=pb_idx,
                        live_time_s=live_time_s,
                        background=st.background,
                    )
                if metric == "var_log_lambda":
                    vals = np.log(lam + eps)
                elif metric == "var_lambda":
                    vals = lam
                else:
                    raise ValueError(f"Unknown surrogate metric: {metric}")
                mean = float(np.sum(weights * vals))
                var = float(np.sum(weights * (vals - mean) ** 2))
                score += alphas.get(iso, 0.0) * var
            scores[oid] = score
    return scores


def _select_candidate_ids(
    scores: Dict[int, float],
    delta: float,
    k_min: int | None,
    k_max: int | None,
) -> List[int]:
    """Select candidate ids by relative threshold with min/max bounds."""
    if not scores:
        return []
    max_score = max(scores.values())
    threshold = (1.0 - delta) * max_score
    candidates = [oid for oid, score in scores.items() if score >= threshold]
    ordered = sorted(scores.keys(), key=lambda oid: scores[oid], reverse=True)
    if k_min is not None and len(candidates) < k_min:
        candidates = ordered[:k_min]
    if k_max is not None and len(candidates) > k_max:
        candidates = ordered[:k_max]
    return candidates


def _eig_scores(
    estimator: RotatingShieldPFEstimator,
    pose_idx: int,
    live_time_s: float,
    candidate_ids: List[int],
    RFe_candidates: np.ndarray,
    RPb_candidates: np.ndarray,
    alpha_by_isotope: Dict[str, float] | None,
    particles_by_isotope: Dict[str, Tuple[list, np.ndarray]] | None,
    num_samples: int,
) -> Dict[int, float]:
    """Compute EIG scores for a list of orientation ids."""
    scores: Dict[int, float] = {}
    num_pb = len(RPb_candidates)
    for oid in candidate_ids:
        fe_idx = oid // num_pb
        pb_idx = oid % num_pb
        score = estimator.orientation_expected_information_gain(
            pose_idx=pose_idx,
            RFe=RFe_candidates[fe_idx],
            RPb=RPb_candidates[pb_idx],
            live_time_s=live_time_s,
            num_samples=num_samples,
            alpha_by_isotope=alpha_by_isotope,
            particles_by_isotope=particles_by_isotope,
        )
        scores[oid] = score
    return scores


def _fisher_scores(
    estimator: RotatingShieldPFEstimator,
    pose_idx: int,
    live_time_s: float,
    candidate_ids: List[int],
    RFe_candidates: np.ndarray,
    RPb_candidates: np.ndarray,
    beta_by_isotope: Dict[str, float] | None,
    particles_by_isotope: Dict[str, Tuple[list, np.ndarray]] | None,
    metric: str,
) -> Dict[int, float]:
    """Compute Fisher (JA/JD) scores for a list of orientation ids."""
    scores: Dict[int, float] = {}
    num_pb = len(RPb_candidates)
    for oid in candidate_ids:
        fe_idx = oid // num_pb
        pb_idx = oid % num_pb
        JA, JD = estimator.orientation_fisher_criteria(
            pose_idx=pose_idx,
            RFe=RFe_candidates[fe_idx],
            RPb=RPb_candidates[pb_idx],
            live_time_s=live_time_s,
            beta_by_isotope=beta_by_isotope,
            particles_by_isotope=particles_by_isotope,
        )
        scores[oid] = JA if metric == "ja" else JD
    return scores


def _eig_scores_with_racing(
    estimator: RotatingShieldPFEstimator,
    pose_idx: int,
    live_time_s: float,
    candidate_ids: List[int],
    RFe_candidates: np.ndarray,
    RPb_candidates: np.ndarray,
    alpha_by_isotope: Dict[str, float] | None,
    particles_by_isotope: Dict[str, Tuple[list, np.ndarray]] | None,
    steps: List[int],
    margin: float,
) -> Dict[int, float]:
    """Compute EIG scores with a simple racing schedule."""
    scores: Dict[int, float] = {}
    eps = 1e-12
    for step in steps:
        scores = _eig_scores(
            estimator=estimator,
            pose_idx=pose_idx,
            live_time_s=live_time_s,
            candidate_ids=candidate_ids,
            RFe_candidates=RFe_candidates,
            RPb_candidates=RPb_candidates,
            alpha_by_isotope=alpha_by_isotope,
            particles_by_isotope=particles_by_isotope,
            num_samples=step,
        )
        ordered = sorted(scores.values(), reverse=True)
        if len(ordered) < 2:
            break
        gap = (ordered[0] - ordered[1]) / max(abs(ordered[0]), eps)
        if gap >= margin:
            break
    return scores


def select_best_orientation(
    estimator: RotatingShieldPFEstimator,
    pose_idx: int,
    live_time_s: float = 1.0,
    fisher_weight: float = 0.0,
    criterion: str | None = None,
    RFe_candidates=None,
    RPb_candidates=None,
    alpha_by_isotope=None,
    beta_by_isotope=None,
    allowed_indices=None,
    eig_samples: int | None = None,
    planning_particles: int | None = None,
    planning_method: str | None = None,
    fisher_metric: str | None = None,
    fisher_screening_k: int | None = None,
    hybrid_eig_samples: int | None = None,
    hybrid_entropy_threshold: float | None = None,
) -> Tuple[int, float]:
    """
    Choose the shield orientation that maximises EIG/JA/JD/variance or hybrid scores.

    Returns:
        (best_orient_idx, best_score) where score = IG + fisher_weight * Fisher
    """
    criterion = estimator.pf_config.orientation_selection_mode if criterion is None else criterion
    if criterion == "fisher":
        criterion = estimator.pf_config.fisher_screening_metric if fisher_metric is None else fisher_metric
        if criterion not in {"ja", "jd"}:
            raise ValueError(f"Unknown Fisher metric: {criterion}")
    if criterion == "fisher":
        criterion = estimator.pf_config.fisher_screening_metric if fisher_metric is None else fisher_metric
        if criterion not in {"ja", "jd"}:
            raise ValueError(f"Unknown Fisher metric: {criterion}")
    scores: List[float] = []
    ids: List[int] = []
    if criterion == "variance":
        for orient_idx in range(estimator.num_orientations):
            ig, fisher = estimator.orientation_information_metrics(
                pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
            )
            scores.append(ig + fisher_weight * fisher)
            ids.append(orient_idx)
    elif criterion in {"eig", "ja", "jd", "hybrid"}:
        if RFe_candidates is None or RPb_candidates is None:
            from measurement.shielding import generate_octant_rotation_matrices
            RFe_candidates = generate_octant_rotation_matrices()
            RPb_candidates = generate_octant_rotation_matrices()
        # Build full Cartesian product so Fe/Pb can point independently (8x8=64 combos by default).
        allowed = set(allowed_indices) if allowed_indices is not None else None
        particles_by_iso = estimator.planning_particles(
            max_particles=planning_particles,
            method=planning_method,
        )
        if criterion == "hybrid":
            fisher_metric = estimator.pf_config.fisher_screening_metric if fisher_metric is None else fisher_metric
            fisher_screening_k = (
                estimator.pf_config.fisher_screening_k if fisher_screening_k is None else fisher_screening_k
            )
            hybrid_eig_samples = (
                estimator.pf_config.hybrid_eig_samples if hybrid_eig_samples is None else hybrid_eig_samples
            )
            hybrid_entropy_threshold = (
                estimator.pf_config.hybrid_entropy_threshold
                if hybrid_entropy_threshold is None
                else hybrid_entropy_threshold
            )
            if fisher_metric not in {"ja", "jd"}:
                raise ValueError(f"Unknown Fisher metric: {fisher_metric}")
            candidate_ids = []
            for fe_idx in range(len(RFe_candidates)):
                for pb_idx in range(len(RPb_candidates)):
                    oid = fe_idx * len(RPb_candidates) + pb_idx
                    if allowed is not None and oid not in allowed:
                        continue
                    candidate_ids.append(oid)
            fisher_scores = _fisher_scores(
                estimator=estimator,
                pose_idx=pose_idx,
                live_time_s=live_time_s,
                candidate_ids=candidate_ids,
                RFe_candidates=RFe_candidates,
                RPb_candidates=RPb_candidates,
                beta_by_isotope=beta_by_isotope,
                particles_by_isotope=particles_by_iso,
                metric=fisher_metric,
            )
            if not fisher_scores:
                return -1, 0.0
            entropy_ratio = estimator.weight_entropy_ratio(particles_by_iso)
            use_eig = entropy_ratio >= hybrid_entropy_threshold
            if use_eig:
                ordered = sorted(fisher_scores.keys(), key=lambda oid: fisher_scores[oid], reverse=True)
                screen_k = max(int(fisher_screening_k), 1)
                screen_k = min(screen_k, len(ordered))
                candidate_ids = ordered[:screen_k]
                eig_scores = _eig_scores(
                    estimator=estimator,
                    pose_idx=pose_idx,
                    live_time_s=live_time_s,
                    candidate_ids=candidate_ids,
                    RFe_candidates=RFe_candidates,
                    RPb_candidates=RPb_candidates,
                    alpha_by_isotope=alpha_by_isotope,
                    particles_by_isotope=particles_by_iso,
                    num_samples=int(hybrid_eig_samples),
                )
                best_id = max(eig_scores.keys(), key=lambda oid: eig_scores[oid])
                return best_id, float(eig_scores[best_id])
            best_id = max(fisher_scores.keys(), key=lambda oid: fisher_scores[oid])
            return best_id, float(fisher_scores[best_id])
        for fe_idx, RFe in enumerate(RFe_candidates):
            for pb_idx, RPb in enumerate(RPb_candidates):
                oid = fe_idx * len(RPb_candidates) + pb_idx
                if allowed is not None and oid not in allowed:
                    continue
                if criterion == "eig":
                    score = estimator.orientation_expected_information_gain(
                        pose_idx=pose_idx,
                        RFe=RFe,
                        RPb=RPb,
                        live_time_s=live_time_s,
                        alpha_by_isotope=alpha_by_isotope,
                        num_samples=eig_samples,
                        particles_by_isotope=particles_by_iso,
                    )
                else:
                    JA, JD = estimator.orientation_fisher_criteria(
                        pose_idx=pose_idx,
                        RFe=RFe,
                        RPb=RPb,
                        live_time_s=live_time_s,
                        beta_by_isotope=beta_by_isotope,
                        particles_by_isotope=particles_by_iso,
                    )
                    score = JA if criterion == "ja" else JD
                scores.append(score)
                ids.append(oid)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    best_idx = int(np.argmax(scores)) if scores else -1
    return ids[best_idx], float(scores[best_idx] if scores else 0.0)


def select_top_k_orientations(
    estimator: RotatingShieldPFEstimator,
    pose_idx: int,
    k: int | None = None,
    live_time_s: float = 1.0,
    fisher_weight: float = 0.0,
    criterion: str | None = None,
    RFe_candidates=None,
    RPb_candidates=None,
    alpha_by_isotope=None,
    beta_by_isotope=None,
    allowed_indices=None,
    eig_samples: int | None = None,
    planning_particles: int | None = None,
    planning_method: str | None = None,
    preselect: bool | None = None,
    preselect_metric: str | None = None,
    preselect_delta: float | None = None,
    preselect_k_min: int | None = None,
    preselect_k_max: int | None = None,
    eig_racing_steps: List[int] | None = None,
    eig_racing_margin: float | None = None,
    fisher_metric: str | None = None,
    fisher_screening_k: int | None = None,
    hybrid_eig_samples: int | None = None,
    hybrid_entropy_threshold: float | None = None,
) -> List[int]:
    """
    Return the top-k orientation ids (Fe/Pb pairs) sorted by score (no replacement).

    This is useful for running multiple short measurements at one pose without repeating
    the same Fe/Pb pair. Uses the same scoring rules as select_best_orientation.
    """
    scores: List[float] = []
    ids: List[int] = []
    k = estimator.pf_config.orientation_k if k is None else k
    criterion = estimator.pf_config.orientation_selection_mode if criterion is None else criterion
    # Reuse select_best_orientation machinery with allowed filtering
    if criterion == "variance":
        for orient_idx in range(estimator.num_orientations):
            if allowed_indices is not None and orient_idx not in set(allowed_indices):
                continue
            ig, fisher = estimator.orientation_information_metrics(
                pose_idx=pose_idx, orient_idx=orient_idx, live_time_s=live_time_s
            )
            scores.append(ig + fisher_weight * fisher)
            ids.append(orient_idx)
    else:
        if RFe_candidates is None or RPb_candidates is None:
            from measurement.shielding import generate_octant_rotation_matrices
            RFe_candidates = generate_octant_rotation_matrices()
            RPb_candidates = generate_octant_rotation_matrices()
        allowed = set(allowed_indices) if allowed_indices is not None else None
        eig_samples = estimator.pf_config.eig_num_samples if eig_samples is None else eig_samples
        planning_particles = (
            estimator.pf_config.planning_particles if planning_particles is None else planning_particles
        )
        planning_method = estimator.pf_config.planning_method if planning_method is None else planning_method
        particles_by_iso = estimator.planning_particles(
            max_particles=planning_particles,
            method=planning_method,
        )
        candidate_ids: List[int] = []
        for fe_idx in range(len(RFe_candidates)):
            for pb_idx in range(len(RPb_candidates)):
                oid = fe_idx * len(RPb_candidates) + pb_idx
                if allowed is not None and oid not in allowed:
                    continue
                candidate_ids.append(oid)

        if criterion == "hybrid":
            fisher_metric = estimator.pf_config.fisher_screening_metric if fisher_metric is None else fisher_metric
            fisher_screening_k = (
                estimator.pf_config.fisher_screening_k if fisher_screening_k is None else fisher_screening_k
            )
            hybrid_eig_samples = (
                estimator.pf_config.hybrid_eig_samples if hybrid_eig_samples is None else hybrid_eig_samples
            )
            hybrid_entropy_threshold = (
                estimator.pf_config.hybrid_entropy_threshold
                if hybrid_entropy_threshold is None
                else hybrid_entropy_threshold
            )
            if fisher_metric not in {"ja", "jd"}:
                raise ValueError(f"Unknown Fisher metric: {fisher_metric}")
            fisher_scores = _fisher_scores(
                estimator=estimator,
                pose_idx=pose_idx,
                live_time_s=live_time_s,
                candidate_ids=candidate_ids,
                RFe_candidates=RFe_candidates,
                RPb_candidates=RPb_candidates,
                beta_by_isotope=beta_by_isotope,
                particles_by_isotope=particles_by_iso,
                metric=fisher_metric,
            )
            if not fisher_scores:
                return []
            entropy_ratio = estimator.weight_entropy_ratio(particles_by_iso)
            use_eig = entropy_ratio >= hybrid_entropy_threshold
            ordered = sorted(fisher_scores.keys(), key=lambda oid: fisher_scores[oid], reverse=True)
            screen_k = max(int(fisher_screening_k), int(k))
            screen_k = min(screen_k, len(ordered))
            screened_ids = ordered[:screen_k]
            if use_eig:
                eig_scores = _eig_scores(
                    estimator=estimator,
                    pose_idx=pose_idx,
                    live_time_s=live_time_s,
                    candidate_ids=screened_ids,
                    RFe_candidates=RFe_candidates,
                    RPb_candidates=RPb_candidates,
                    alpha_by_isotope=alpha_by_isotope,
                    particles_by_isotope=particles_by_iso,
                    num_samples=int(hybrid_eig_samples),
                )
                ids = screened_ids
                scores = [eig_scores[oid] for oid in screened_ids]
            else:
                ids = screened_ids
                scores = [fisher_scores[oid] for oid in screened_ids]
        else:
            preselect = estimator.pf_config.preselect_orientations if preselect is None else preselect
            preselect_metric = estimator.pf_config.preselect_metric if preselect_metric is None else preselect_metric
            preselect_delta = estimator.pf_config.preselect_delta if preselect_delta is None else preselect_delta
            preselect_k_min = estimator.pf_config.preselect_k_min if preselect_k_min is None else preselect_k_min
            preselect_k_max = estimator.pf_config.preselect_k_max if preselect_k_max is None else preselect_k_max
            eig_racing_margin = (
                estimator.pf_config.preselect_delta if eig_racing_margin is None else eig_racing_margin
            )
            if eig_racing_steps is None:
                eig_racing_steps = [eig_samples]

            if criterion == "eig" and preselect:
                surrogate_scores = _surrogate_scores(
                    estimator=estimator,
                    pose_idx=pose_idx,
                    live_time_s=live_time_s,
                    particles_by_isotope=particles_by_iso,
                    RFe_candidates=RFe_candidates,
                    RPb_candidates=RPb_candidates,
                    alpha_by_isotope=alpha_by_isotope,
                    allowed_indices=allowed,
                    metric=preselect_metric,
                )
                candidate_ids = _select_candidate_ids(
                    scores=surrogate_scores,
                    delta=preselect_delta,
                    k_min=preselect_k_min,
                    k_max=preselect_k_max,
                )

            for fe_idx, RFe in enumerate(RFe_candidates):
                for pb_idx, RPb in enumerate(RPb_candidates):
                    oid = fe_idx * len(RPb_candidates) + pb_idx
                    if oid not in candidate_ids:
                        continue
                    if criterion == "eig":
                        if len(eig_racing_steps) > 1:
                            scores_dict = _eig_scores_with_racing(
                                estimator=estimator,
                                pose_idx=pose_idx,
                                live_time_s=live_time_s,
                                candidate_ids=candidate_ids,
                                RFe_candidates=RFe_candidates,
                                RPb_candidates=RPb_candidates,
                                alpha_by_isotope=alpha_by_isotope,
                                particles_by_isotope=particles_by_iso,
                                steps=eig_racing_steps,
                                margin=eig_racing_margin,
                            )
                            scores = [scores_dict[oid] for oid in candidate_ids]
                            ids = candidate_ids
                            break
                        score = estimator.orientation_expected_information_gain(
                            pose_idx=pose_idx,
                            RFe=RFe,
                            RPb=RPb,
                            live_time_s=live_time_s,
                            alpha_by_isotope=alpha_by_isotope,
                            num_samples=eig_samples,
                            particles_by_isotope=particles_by_iso,
                        )
                    else:
                        JA, JD = estimator.orientation_fisher_criteria(
                            pose_idx=pose_idx,
                            RFe=RFe,
                            RPb=RPb,
                            live_time_s=live_time_s,
                            beta_by_isotope=beta_by_isotope,
                            particles_by_isotope=particles_by_iso,
                        )
                        score = JA if criterion == "ja" else JD
                    scores.append(score)
                    ids.append(oid)
                if len(eig_racing_steps) > 1 and criterion == "eig" and ids:
                    break
    if not scores:
        return []
    order = np.argsort(scores)[::-1]
    top_ids = [ids[i] for i in order[:k]]
    return top_ids


def rotation_policy_step(
    estimator: RotatingShieldPFEstimator,
    pose_idx: int,
    ig_threshold: float = 1e-3,
    fisher_threshold: float = 1e-3,
    live_time_s: float = 0.5,
    fisher_weight: float = 0.0,
) -> Tuple[bool, int, float]:
    """
    One step of the shield-rotation policy (Sec. 3.4.3, Eqs. 3.47–3.48).

    - Compute expected IG/Fisher for all orientations at the current pose using
      the discrete particle states when available.
    - If max metrics are below thresholds, stop rotating.
    - Otherwise select the best orientation (short acquisition suggested by live_time_s).

    Returns:
        (should_stop, orient_idx, score)
    """
    igs: List[float] = []
    fishers: List[float] = []
    scores: List[float] = []
    for oid in range(estimator.num_orientations):
        ig, fi = estimator.orientation_information_metrics(
            pose_idx=pose_idx,
            orient_idx=oid,
            live_time_s=live_time_s,
            prefer_continuous=False,
        )
        igs.append(ig)
        fishers.append(fi)
        scores.append(ig + fisher_weight * fi)
    max_ig = max(igs) if igs else 0.0
    max_fi = max(fishers) if fishers else 0.0
    if (max_ig < ig_threshold) and (max_fi < fisher_threshold):
        return True, -1, 0.0
    best_idx = int(np.argmax(scores))
    return False, best_idx, float(scores[best_idx])
