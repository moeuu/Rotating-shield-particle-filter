"""Shield orientation selection based on information metrics (Sec. 3.4)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from pf.estimator import RotatingShieldPFEstimator


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Return a normalized copy of weights."""
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.ones_like(weights) / max(len(weights), 1)
    return weights / total


def _resolve_gpu_context(
    estimator: RotatingShieldPFEstimator,
) -> Tuple[object, object, object, object] | None:
    """Return (torch, gpu_utils, device, dtype) for GPU evaluation if available."""
    if not hasattr(estimator, "_gpu_enabled") or not estimator._gpu_enabled():
        raise RuntimeError("GPU-only mode requires estimator GPU support.")
    from pf import gpu_utils
    import torch
    device = gpu_utils.resolve_device(estimator.pf_config.gpu_device)
    dtype = gpu_utils.resolve_dtype(estimator.pf_config.gpu_dtype)
    return torch, gpu_utils, device, dtype


def _normalize_weights_torch(weights_t: object, torch_mod: object) -> object:
    """Normalize a torch weight vector with a uniform fallback."""
    weight_sum = torch_mod.sum(weights_t)
    if float(weight_sum) <= 0.0:
        return torch_mod.full_like(weights_t, 1.0 / max(weights_t.numel(), 1))
    return weights_t / weight_sum


def _pack_states_by_isotope(
    particles_by_isotope: Dict[str, Tuple[list, np.ndarray]],
    torch_mod: object,
    gpu_utils: object,
    device: object,
    dtype: object,
) -> Dict[str, Tuple[object, object, object, object, object]]:
    """Pack isotope particle states into GPU tensors with normalized weights."""
    packed: Dict[str, Tuple[object, object, object, object, object]] = {}
    for iso, (states, weights) in particles_by_isotope.items():
        if not states:
            continue
        positions, strengths, backgrounds, mask = gpu_utils.pack_states(states, device=device, dtype=dtype)
        weights_t = torch_mod.as_tensor(np.asarray(weights, dtype=float), device=device, dtype=dtype)
        weights_t = _normalize_weights_torch(weights_t, torch_mod)
        packed[iso] = (positions, strengths, backgrounds, mask, weights_t)
    return packed


def _filter_planning_isotopes(
    planning_isotopes: Sequence[str] | None,
    particles_by_isotope: Dict[str, Tuple[list, np.ndarray]],
    alpha_by_isotope: Dict[str, float] | None,
) -> Tuple[Dict[str, Tuple[list, np.ndarray]], Dict[str, float] | None]:
    """Filter particles/weights to the requested planning isotopes when provided."""
    if planning_isotopes is None:
        return particles_by_isotope, alpha_by_isotope
    planning_set = set(planning_isotopes)
    filtered_particles = {
        iso: val for iso, val in particles_by_isotope.items() if iso in planning_set
    }
    if not filtered_particles:
        return particles_by_isotope, alpha_by_isotope
    if alpha_by_isotope is None:
        filtered_alpha = {iso: 1.0 for iso in planning_set}
    else:
        filtered_alpha = {
            iso: float(alpha_by_isotope.get(iso, 1.0)) for iso in planning_set
        }
    return filtered_particles, filtered_alpha


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
    alphas = alpha_by_isotope or {iso: 1.0 for iso in particles_by_isotope}
    alpha_sum = sum(alphas.values()) or 1.0
    alphas = {k: v / alpha_sum for k, v in alphas.items()}
    eps = 1e-12
    gpu_ctx = _resolve_gpu_context(estimator)
    return _surrogate_scores_gpu(
        estimator=estimator,
        pose_idx=pose_idx,
        live_time_s=live_time_s,
        particles_by_isotope=particles_by_isotope,
        RFe_candidates=RFe_candidates,
        RPb_candidates=RPb_candidates,
        alphas=alphas,
        allowed_indices=allowed_indices,
        metric=metric,
        gpu_ctx=gpu_ctx,
        eps=eps,
    )


def _surrogate_scores_gpu(
    estimator: RotatingShieldPFEstimator,
    pose_idx: int,
    live_time_s: float,
    particles_by_isotope: Dict[str, Tuple[list, np.ndarray]],
    RFe_candidates: np.ndarray,
    RPb_candidates: np.ndarray,
    alphas: Dict[str, float],
    allowed_indices: set[int] | None,
    metric: str,
    gpu_ctx: Tuple[object, object, object, object],
    eps: float,
) -> Dict[int, float]:
    """Compute surrogate scores on GPU using packed particle tensors."""
    if pose_idx < 0 or pose_idx >= len(estimator.poses):
        raise IndexError("pose_idx out of range")
    torch_mod, gpu_utils, device, dtype = gpu_ctx
    from measurement.continuous_kernels import ContinuousKernel

    packed = _pack_states_by_isotope(particles_by_isotope, torch_mod, gpu_utils, device, dtype)
    if not packed:
        return {}
    detector_pos = np.asarray(estimator.poses[pose_idx], dtype=float)
    kernel = ContinuousKernel(mu_by_isotope=estimator.mu_by_isotope, shield_params=estimator.shield_params)
    from measurement.shielding import octant_index_from_rotation

    fe_indices = [octant_index_from_rotation(R) for R in RFe_candidates]
    pb_indices = [octant_index_from_rotation(R) for R in RPb_candidates]
    mu_by_iso = {iso: kernel._mu_values(isotope=iso) for iso in packed}
    shield_params = kernel.shield_params
    scores: Dict[int, float] = {}
    num_pb = len(RPb_candidates)
    for fe_idx in range(len(RFe_candidates)):
        for pb_idx in range(len(RPb_candidates)):
            oid = fe_idx * num_pb + pb_idx
            if allowed_indices is not None and oid not in allowed_indices:
                continue
            score = 0.0
            for iso, (positions, strengths, backgrounds, mask, weights_t) in packed.items():
                mu_fe, mu_pb = mu_by_iso[iso]
                lam_t = gpu_utils.expected_counts_pair_torch(
                    detector_pos=detector_pos,
                    positions=positions,
                    strengths=strengths,
                    backgrounds=backgrounds,
                    mask=mask,
                    fe_index=fe_idx,
                    pb_index=pb_idx,
                    mu_fe=mu_fe,
                    mu_pb=mu_pb,
                    thickness_fe_cm=shield_params.thickness_fe_cm,
                    thickness_pb_cm=shield_params.thickness_pb_cm,
                    use_angle_attenuation=shield_params.use_angle_attenuation,
                    live_time_s=live_time_s,
                    device=device,
                    dtype=dtype,
                    source_scale=estimator.response_scale_for_isotope(iso),
                )
                if metric == "var_log_lambda":
                    vals = torch_mod.log(lam_t + eps)
                elif metric == "var_lambda":
                    vals = lam_t
                else:
                    raise ValueError(f"Unknown surrogate metric: {metric}")
                mean = torch_mod.sum(weights_t * vals)
                var = torch_mod.sum(weights_t * (vals - mean) ** 2)
                score += alphas.get(iso, 0.0) * float(var.item())
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
    num_samples: int | None,
) -> Dict[int, float]:
    """Compute EIG scores for a list of orientation ids."""
    if not candidate_ids:
        return {}
    if num_samples is None:
        num_samples = estimator.pf_config.eig_num_samples
    if not particles_by_isotope:
        raise RuntimeError("GPU-only mode requires particles_by_isotope for EIG scoring.")
    gpu_ctx = _resolve_gpu_context(estimator)
    return _eig_scores_gpu(
        estimator=estimator,
        pose_idx=pose_idx,
        live_time_s=live_time_s,
        candidate_ids=candidate_ids,
        RFe_candidates=RFe_candidates,
        RPb_candidates=RPb_candidates,
        alpha_by_isotope=alpha_by_isotope,
        particles_by_isotope=particles_by_isotope,
        num_samples=num_samples,
        gpu_ctx=gpu_ctx,
    )


def _eig_scores_gpu(
    estimator: RotatingShieldPFEstimator,
    pose_idx: int,
    live_time_s: float,
    candidate_ids: List[int],
    RFe_candidates: np.ndarray,
    RPb_candidates: np.ndarray,
    alpha_by_isotope: Dict[str, float] | None,
    particles_by_isotope: Dict[str, Tuple[list, np.ndarray]],
    num_samples: int | None,
    gpu_ctx: Tuple[object, object, object, object],
) -> Dict[int, float]:
    """Compute EIG scores on GPU by reusing packed particle tensors."""
    if pose_idx < 0 or pose_idx >= len(estimator.poses):
        raise IndexError("pose_idx out of range")
    if not candidate_ids:
        return {}
    torch_mod, gpu_utils, device, dtype = gpu_ctx
    from measurement.continuous_kernels import ContinuousKernel

    detector_pos = np.asarray(estimator.poses[pose_idx], dtype=float)
    kernel = ContinuousKernel(mu_by_isotope=estimator.mu_by_isotope, shield_params=estimator.shield_params)
    packed = _pack_states_by_isotope(particles_by_isotope, torch_mod, gpu_utils, device, dtype)
    if not packed:
        return {}
    alphas = alpha_by_isotope or {iso: 1.0 for iso in packed}
    alpha_sum = sum(alphas.values()) or 1.0
    alphas = {k: v / alpha_sum for k, v in alphas.items()}
    eps = 1e-12
    num_pb = len(RPb_candidates)
    if num_samples is None:
        num_samples = estimator.pf_config.eig_num_samples
    num_samples = int(num_samples)
    mu_by_iso = {iso: kernel._mu_values(isotope=iso) for iso in packed}
    shield_params = kernel.shield_params
    cache: Dict[str, Tuple[object, object, object, object, object, object, object, object, object]] = {}
    for iso, (positions, strengths, backgrounds, mask, weights_t) in packed.items():
        log_weights_t = torch_mod.log(weights_t + eps)
        H_prior = -torch_mod.sum(weights_t * log_weights_t)
        mu_fe, mu_pb = mu_by_iso[iso]
        cache[iso] = (positions, strengths, backgrounds, mask, weights_t, log_weights_t, H_prior, mu_fe, mu_pb)

    scores: Dict[int, float] = {}
    for oid in candidate_ids:
        fe_idx = oid // num_pb
        pb_idx = oid % num_pb
        total_ig = 0.0
        for iso, (
            positions,
            strengths,
            backgrounds,
            mask,
            weights_t,
            log_weights_t,
            H_prior,
            mu_fe,
            mu_pb,
        ) in cache.items():
            lam_t = gpu_utils.expected_counts_pair_torch(
                detector_pos=detector_pos,
                positions=positions,
                strengths=strengths,
                backgrounds=backgrounds,
                mask=mask,
                fe_index=fe_idx,
                pb_index=pb_idx,
                mu_fe=mu_fe,
                mu_pb=mu_pb,
                thickness_fe_cm=shield_params.thickness_fe_cm,
                thickness_pb_cm=shield_params.thickness_pb_cm,
                use_angle_attenuation=shield_params.use_angle_attenuation,
                live_time_s=live_time_s,
                device=device,
                dtype=dtype,
                source_scale=estimator.response_scale_for_isotope(iso),
            )
            if num_samples <= 0:
                H_post_mean = torch_mod.zeros((), device=device, dtype=dtype)
            else:
                idx = torch_mod.multinomial(weights_t, num_samples, replacement=True)
                z = torch_mod.poisson(lam_t[idx])
                logw = log_weights_t + z.unsqueeze(1) * torch_mod.log(lam_t + eps) - lam_t
                logw = logw - torch_mod.logsumexp(logw, dim=1, keepdim=True)
                w_post = torch_mod.exp(logw)
                H_post = -torch_mod.sum(w_post * torch_mod.log(w_post + eps), dim=1)
                H_post_mean = torch_mod.mean(H_post)
            ig_h = float((H_prior - H_post_mean).item())
            total_ig += alphas.get(iso, 0.0) * ig_h
        scores[oid] = float(total_ig)
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
    RFe_candidates=None,
    RPb_candidates=None,
    alpha_by_isotope=None,
    planning_isotopes: Sequence[str] | None = None,
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
) -> Tuple[int, float]:
    """
    Choose the shield orientation that maximizes expected information gain (EIG).

    Returns:
        (best_orient_idx, best_score)
    """
    if RFe_candidates is None or RPb_candidates is None:
        from measurement.shielding import generate_octant_rotation_matrices

        RFe_candidates = generate_octant_rotation_matrices()
        RPb_candidates = generate_octant_rotation_matrices()
    allowed = set(allowed_indices) if allowed_indices is not None else None
    particles_by_iso = estimator.planning_particles(
        max_particles=planning_particles,
        method=planning_method,
    )
    particles_by_iso, alpha_by_isotope = _filter_planning_isotopes(
        planning_isotopes,
        particles_by_iso,
        alpha_by_isotope,
    )
    candidate_ids: List[int] = []
    for fe_idx in range(len(RFe_candidates)):
        for pb_idx in range(len(RPb_candidates)):
            oid = fe_idx * len(RPb_candidates) + pb_idx
            if allowed is not None and oid not in allowed:
                continue
            candidate_ids.append(oid)
    if not candidate_ids:
        return -1, 0.0
    preselect = estimator.pf_config.preselect_orientations if preselect is None else preselect
    preselect_metric = estimator.pf_config.preselect_metric if preselect_metric is None else preselect_metric
    preselect_delta = estimator.pf_config.preselect_delta if preselect_delta is None else preselect_delta
    preselect_k_min = estimator.pf_config.preselect_k_min if preselect_k_min is None else preselect_k_min
    preselect_k_max = estimator.pf_config.preselect_k_max if preselect_k_max is None else preselect_k_max
    if preselect:
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
    if not candidate_ids:
        return -1, 0.0
    if eig_racing_steps is None:
        eig_racing_steps = [eig_samples if eig_samples is not None else estimator.pf_config.eig_num_samples]
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
            steps=[int(s) for s in eig_racing_steps],
            margin=0.05 if eig_racing_margin is None else float(eig_racing_margin),
        )
    else:
        scores_dict = _eig_scores(
            estimator=estimator,
            pose_idx=pose_idx,
            live_time_s=live_time_s,
            candidate_ids=candidate_ids,
            RFe_candidates=RFe_candidates,
            RPb_candidates=RPb_candidates,
            alpha_by_isotope=alpha_by_isotope,
            particles_by_isotope=particles_by_iso,
            num_samples=eig_samples,
        )
    if not scores_dict:
        return -1, 0.0
    best_id = max(scores_dict.keys(), key=lambda oid: scores_dict[oid])
    best_score = max(float(scores_dict[best_id]), 0.0)
    return best_id, best_score


def select_top_k_orientations(
    estimator: RotatingShieldPFEstimator,
    pose_idx: int,
    k: int | None = None,
    live_time_s: float = 1.0,
    RFe_candidates=None,
    RPb_candidates=None,
    alpha_by_isotope=None,
    planning_isotopes: Sequence[str] | None = None,
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
) -> List[int]:
    """
    Return the top-k orientation ids (Fe/Pb pairs) sorted by EIG (no replacement).
    """
    k = estimator.pf_config.orientation_k if k is None else k
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
    particles_by_iso, alpha_by_isotope = _filter_planning_isotopes(
        planning_isotopes,
        particles_by_iso,
        alpha_by_isotope,
    )
    candidate_ids: List[int] = []
    for fe_idx in range(len(RFe_candidates)):
        for pb_idx in range(len(RPb_candidates)):
            oid = fe_idx * len(RPb_candidates) + pb_idx
            if allowed is not None and oid not in allowed:
                continue
            candidate_ids.append(oid)
    if not candidate_ids:
        return []
    preselect = estimator.pf_config.preselect_orientations if preselect is None else preselect
    preselect_metric = estimator.pf_config.preselect_metric if preselect_metric is None else preselect_metric
    preselect_delta = estimator.pf_config.preselect_delta if preselect_delta is None else preselect_delta
    preselect_k_min = estimator.pf_config.preselect_k_min if preselect_k_min is None else preselect_k_min
    preselect_k_max = estimator.pf_config.preselect_k_max if preselect_k_max is None else preselect_k_max
    eig_racing_margin = estimator.pf_config.preselect_delta if eig_racing_margin is None else eig_racing_margin
    if eig_racing_steps is None:
        eig_racing_steps = [eig_samples]
    if preselect:
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
    if not candidate_ids:
        return []
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
    else:
        scores_dict = _eig_scores(
            estimator=estimator,
            pose_idx=pose_idx,
            live_time_s=live_time_s,
            candidate_ids=candidate_ids,
            RFe_candidates=RFe_candidates,
            RPb_candidates=RPb_candidates,
            alpha_by_isotope=alpha_by_isotope,
            particles_by_isotope=particles_by_iso,
            num_samples=eig_samples,
        )
    if not scores_dict:
        return []
    scores = [scores_dict[oid] for oid in candidate_ids]
    order = np.argsort(scores)[::-1]
    top_ids = [candidate_ids[i] for i in order[:k]]
    return top_ids


def rotation_policy_step(
    estimator: RotatingShieldPFEstimator,
    pose_idx: int,
    ig_threshold: float = 1e-3,
    live_time_s: float = 0.5,
) -> Tuple[bool, int, float]:
    """
    One step of the shield-rotation policy (Sec. 3.4.3, Eqs. 3.47–3.48).

    - Compute expected IG for all orientations at the current pose using
      the continuous particle states.
    - If max IG is below the threshold, stop rotating.
    - Otherwise select the best orientation (short acquisition suggested by live_time_s).

    Returns:
        (should_stop, orient_idx, score)
    """
    igs: List[float] = []
    scores: List[float] = []
    for oid in range(estimator.num_orientations):
        ig = estimator.orientation_information_gain(
            pose_idx=pose_idx,
            orient_idx=oid,
            live_time_s=live_time_s,
        )
        igs.append(ig)
        scores.append(ig)
    max_ig = max(igs) if igs else 0.0
    if max_ig < ig_threshold:
        return True, -1, 0.0
    best_idx = int(np.argmax(scores))
    return False, best_idx, float(scores[best_idx])
