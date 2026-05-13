"""JAX implementation of the offline transport drift loss.

This ports the active PyTorch loss used by the supplied drifting actor runner:
``offline_transport_drift_loss``.  The legacy heuristic loss from that file is
not included here because the OGBench runner uses the score-difference
transport objective.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _pairwise_sqdist(x, y):
    """Pairwise squared L2 distance: [B, N, D] x [B, M, D] -> [B, N, M]."""
    xydot = jnp.einsum("bnd,bmd->bnm", x, y)
    xnorms = jnp.einsum("bnd,bnd->bn", x, x)
    ynorms = jnp.einsum("bmd,bmd->bm", y, y)
    return jnp.clip(xnorms[:, :, None] + ynorms[:, None, :] - 2.0 * xydot, min=0.0)


def _format_bandwidth(bandwidth, query, refs, eps=1e-8):
    bandwidth = jnp.asarray(bandwidth, dtype=query.dtype)

    if bandwidth.ndim == 0:
        return jnp.clip(bandwidth, min=eps)
    if bandwidth.ndim == 1:
        if bandwidth.shape[0] == query.shape[0]:
            bandwidth = bandwidth[:, None, None]
        elif bandwidth.shape[0] == query.shape[1]:
            bandwidth = bandwidth[None, :, None]
        else:
            raise ValueError(
                "1D bandwidth must match batch or query-particle count, got "
                f"{bandwidth.shape} for query {query.shape}"
            )
    elif bandwidth.ndim == 2:
        if bandwidth.shape != query.shape[:2]:
            raise ValueError(
                "2D bandwidth must have shape [B, N], got "
                f"{bandwidth.shape} for query {query.shape}"
            )
        bandwidth = bandwidth[..., None]
    elif bandwidth.ndim == 3:
        if bandwidth.shape[:2] != query.shape[:2]:
            raise ValueError(
                "3D bandwidth must start with [B, N], got "
                f"{bandwidth.shape} for query {query.shape}"
            )
    else:
        raise ValueError(f"Unsupported bandwidth shape: {bandwidth.shape}")

    return jnp.clip(bandwidth, min=eps)


def _adaptive_bandwidth(
    query,
    refs,
    fallback_bandwidth,
    quantile=0.5,
    scale=1.0,
    min_bandwidth=1e-3,
    max_bandwidth=None,
    exclude_self=False,
    eps=1e-8,
):
    fallback = _format_bandwidth(fallback_bandwidth, query, refs, eps=eps)
    dist = jnp.sqrt(jnp.clip(_pairwise_sqdist(query, refs), min=eps))
    valid_count = refs.shape[1]

    if exclude_self:
        if query.shape[1] != refs.shape[1]:
            raise ValueError("exclude_self=True requires aligned query/ref particle counts")
        diag = jnp.eye(query.shape[1], dtype=bool)
        dist = jnp.where(diag[None, :, :], jnp.inf, dist)
        valid_count -= 1

    if valid_count <= 0:
        return fallback

    quantile = max(0.0, min(1.0, float(quantile)))
    kth = int(round(quantile * (valid_count - 1)))
    bandwidth = jnp.sort(dist, axis=-1)[..., kth] * float(scale)
    fallback_values = fallback
    if fallback_values.ndim == 0:
        fallback_values = jnp.broadcast_to(fallback_values, bandwidth.shape)
    else:
        fallback_values = jnp.broadcast_to(jnp.squeeze(fallback_values, axis=-1), bandwidth.shape)
    bandwidth = jnp.where(jnp.isfinite(bandwidth), bandwidth, fallback_values)
    bandwidth = jnp.clip(bandwidth, min=float(min_bandwidth))
    if max_bandwidth is not None:
        bandwidth = jnp.clip(bandwidth, max=float(max_bandwidth))
    return bandwidth[..., None]


def _resolve_bandwidth(
    query,
    refs,
    bandwidth,
    adaptive=False,
    quantile=0.5,
    scale=1.0,
    min_bandwidth=1e-3,
    max_bandwidth=None,
    exclude_self=False,
):
    if adaptive:
        return _adaptive_bandwidth(
            query,
            refs,
            bandwidth,
            quantile=quantile,
            scale=scale,
            min_bandwidth=min_bandwidth,
            max_bandwidth=max_bandwidth,
            exclude_self=exclude_self,
        )
    return _format_bandwidth(bandwidth, query, refs)


def _clip_score_norm(score, max_norm):
    if max_norm is None or max_norm <= 0:
        return score
    norm = jnp.clip(jnp.linalg.norm(score, axis=-1, keepdims=True), min=1e-8)
    scale = jnp.minimum(float(max_norm) / norm, 1.0)
    return score * scale


def _norm_quantile(value, quantile):
    norm = jnp.linalg.norm(value.reshape((-1, value.shape[-1])), axis=-1)
    if norm.shape[0] <= 1:
        return jnp.max(norm)
    idx = int(round(float(quantile) * (norm.shape[0] - 1)))
    return jnp.sort(norm)[idx]


def _rms(value):
    return jnp.sqrt(jnp.maximum(jnp.mean(jnp.square(value)), 0.0))


def _safe_cos(a, b):
    a = a.reshape((-1, a.shape[-1]))
    b = b.reshape((-1, b.shape[-1]))
    denom = jnp.sqrt(jnp.sum(jnp.square(a)) * jnp.sum(jnp.square(b)))
    return jnp.where(denom > 1e-12, jnp.sum(a * b) / denom, jnp.asarray(0.0, dtype=a.dtype))


def _gaussian_kde_score(query, refs, bandwidth, ref_weights=None, exclude_self=False, eps=1e-8):
    """Gaussian-KDE score grad_query log p(query)."""
    if query.ndim != 3 or refs.ndim != 3:
        raise ValueError("query and refs must have shape [B, N, S] and [B, M, S]")
    if query.shape[0] != refs.shape[0] or query.shape[-1] != refs.shape[-1]:
        raise ValueError(
            "query and refs must share batch and sample dimensions, got "
            f"{query.shape} and {refs.shape}"
        )
    if exclude_self and query.shape[1] != refs.shape[1]:
        raise ValueError("exclude_self=True requires aligned query/ref particle counts")

    bandwidth = _format_bandwidth(bandwidth, query, refs, eps=eps)
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    diff = refs[:, None, :, :] - query[:, :, None, :]
    logits = -0.5 * _pairwise_sqdist(query, refs) * inv_h2

    if exclude_self:
        diag = jnp.eye(query.shape[1], dtype=bool)
        logits = jnp.where(diag[None, :, :], -jnp.inf, logits)

    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    safe_max_logits = jnp.where(jnp.isfinite(max_logits), max_logits, jnp.zeros_like(max_logits))
    shifted_logits = jnp.where(jnp.isfinite(logits), logits - safe_max_logits, logits)
    weights = jnp.exp(shifted_logits)

    if ref_weights is not None:
        ref_weights = jnp.asarray(ref_weights, dtype=query.dtype)
        weights = weights * jnp.clip(ref_weights[:, None, :], min=0.0)
    if exclude_self:
        weights = jnp.where(jnp.isfinite(logits), weights, jnp.zeros_like(weights))

    denom = jnp.clip(jnp.sum(weights, axis=-1, keepdims=True), min=eps)
    return jnp.sum(weights[..., None] * diff, axis=2) / denom * inv_h2


def _compact_kernel_terms(u2, diff, inv_h2, kernel, eps):
    inside = u2 < 1.0
    inside_f = inside.astype(diff.dtype)
    safe_gap = jnp.clip(1.0 - u2, min=eps)

    if kernel == "epanechnikov":
        density = jnp.where(inside, safe_gap, 0.0)
        grad = jnp.where(inside[..., None], 2.0 * diff * inv_h2[..., None], 0.0)
    elif kernel == "bump":
        density = jnp.where(inside, jnp.exp(-1.0 / safe_gap), 0.0)
        grad_log = 2.0 * diff * inv_h2[..., None] / jnp.square(safe_gap[..., None])
        grad = density[..., None] * jnp.where(inside[..., None], grad_log, 0.0)
    elif kernel == "tricube":
        u = jnp.sqrt(jnp.clip(u2, min=0.0))
        one_minus_u3 = jnp.clip(1.0 - u * u * u, min=0.0)
        density = jnp.where(inside, one_minus_u3**3, 0.0)
        grad_coef = 9.0 * u * jnp.square(one_minus_u3) * inv_h2
        grad = jnp.where(inside[..., None], grad_coef[..., None] * diff, 0.0)
    else:
        raise ValueError(f"Unsupported compact KDE kernel: {kernel}")
    return density, grad, inside_f


def _compact_kde_score(
    query,
    refs,
    bandwidth,
    fallback_bandwidth=None,
    ref_weights=None,
    exclude_self=False,
    kernel="epanechnikov",
    compact_eps=1e-6,
    score_clip=100.0,
    min_neighbors=1,
    fallback="nearest",
    eps=1e-8,
):
    if query.ndim != 3 or refs.ndim != 3:
        raise ValueError("query and refs must have shape [B, N, S] and [B, M, S]")
    if query.shape[0] != refs.shape[0] or query.shape[-1] != refs.shape[-1]:
        raise ValueError(
            "query and refs must share batch and sample dimensions, got "
            f"{query.shape} and {refs.shape}"
        )
    if exclude_self and query.shape[1] != refs.shape[1]:
        raise ValueError("exclude_self=True requires aligned query/ref particle counts")

    bandwidth = _format_bandwidth(bandwidth, query, refs, eps=eps)
    if fallback_bandwidth is None:
        fallback_bandwidth = bandwidth
    else:
        fallback_bandwidth = _format_bandwidth(fallback_bandwidth, query, refs, eps=eps)
    inv_h2 = 1.0 / (bandwidth * bandwidth)
    fallback_inv_h2 = 1.0 / (fallback_bandwidth * fallback_bandwidth)
    diff = refs[:, None, :, :] - query[:, :, None, :]
    sqdist = _pairwise_sqdist(query, refs)
    u2 = sqdist * inv_h2
    valid = jnp.ones_like(sqdist, dtype=bool)
    if exclude_self:
        diag = jnp.eye(query.shape[1], dtype=bool)
        valid = jnp.logical_not(diag[None, :, :])

    ref_weights_for_terms = None
    if ref_weights is not None:
        ref_weights_for_terms = jnp.asarray(ref_weights, dtype=query.dtype)
        ref_weights_for_terms = jnp.clip(ref_weights_for_terms[:, None, :], min=0.0)
        valid = jnp.logical_and(valid, ref_weights_for_terms > 0.0)

    density_terms, grad_terms, inside_f = _compact_kernel_terms(
        u2,
        diff,
        inv_h2,
        kernel,
        compact_eps,
    )
    density_terms = jnp.where(valid, density_terms, 0.0)
    grad_terms = jnp.where(valid[..., None], grad_terms, 0.0)
    inside_f = jnp.where(valid, inside_f, 0.0)

    if ref_weights_for_terms is not None:
        density_terms = density_terms * ref_weights_for_terms
        grad_terms = grad_terms * ref_weights_for_terms[..., None]

    density = jnp.sum(density_terms, axis=-1, keepdims=True)
    numerator = jnp.sum(grad_terms, axis=2)
    score = numerator / jnp.clip(density, min=float(compact_eps))

    inside_count = jnp.sum(inside_f, axis=-1)
    no_neighbor = jnp.logical_or(
        inside_count < float(min_neighbors),
        jnp.squeeze(density, axis=-1) <= float(compact_eps),
    )

    if fallback == "gaussian":
        fallback_score = _gaussian_kde_score(
            query,
            refs,
            fallback_bandwidth,
            ref_weights=ref_weights_for_terms[:, 0, :] if ref_weights_for_terms is not None else None,
            exclude_self=exclude_self,
            eps=eps,
        )
    elif fallback == "nearest":
        masked_sqdist = jnp.where(valid, sqdist, jnp.inf)
        nearest = jnp.argmin(masked_sqdist, axis=-1)
        nearest_ref = jnp.take_along_axis(
            refs[:, None, :, :],
            nearest[..., None, None],
            axis=2,
        )[:, :, 0, :]
        fallback_score = (nearest_ref - query) * fallback_inv_h2
    elif fallback == "zero":
        fallback_score = jnp.zeros_like(query)
    else:
        raise ValueError(f"Unsupported compact KDE fallback: {fallback}")

    score = jnp.where(no_neighbor[..., None], fallback_score, score)
    score = _clip_score_norm(score, score_clip)
    info = {
        "compact_no_neighbor_fraction": jnp.mean(no_neighbor.astype(query.dtype)),
        "compact_inside_neighbor_count_mean": jnp.mean(inside_count),
        "compact_inside_neighbor_count_min": jnp.min(inside_count),
        "compact_density_mean": jnp.mean(density),
        "compact_density_min": jnp.min(density),
    }
    return score, info


def _kde_score(
    query,
    refs,
    bandwidth,
    fallback_bandwidth=None,
    ref_weights=None,
    exclude_self=False,
    eps=1e-8,
    kernel="gaussian",
    compact_eps=1e-6,
    compact_score_clip=100.0,
    compact_min_neighbors=1,
    compact_fallback="nearest",
    return_info=False,
):
    if kernel == "gaussian":
        score = _gaussian_kde_score(
            query,
            refs,
            bandwidth,
            ref_weights=ref_weights,
            exclude_self=exclude_self,
            eps=eps,
        )
        info = {
            "compact_no_neighbor_fraction": jnp.asarray(0.0, dtype=query.dtype),
            "compact_inside_neighbor_count_mean": jnp.asarray(float(refs.shape[1]), dtype=query.dtype),
            "compact_inside_neighbor_count_min": jnp.asarray(float(refs.shape[1]), dtype=query.dtype),
            "compact_density_mean": jnp.asarray(0.0, dtype=query.dtype),
            "compact_density_min": jnp.asarray(0.0, dtype=query.dtype),
        }
    else:
        score, info = _compact_kde_score(
            query,
            refs,
            bandwidth,
            fallback_bandwidth=fallback_bandwidth,
            ref_weights=ref_weights,
            exclude_self=exclude_self,
            kernel=kernel,
            compact_eps=compact_eps,
            score_clip=compact_score_clip,
            min_neighbors=compact_min_neighbors,
            fallback=compact_fallback,
            eps=eps,
        )
    if return_info:
        return score, info
    return score


def _as_particle_set(value, ref, batch_size, sample_dim, name):
    if value is None:
        return None
    value = jnp.asarray(value, dtype=ref.dtype)
    if value.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch dimension {value.shape[0]} does not match gen batch {batch_size}"
        )
    if value.ndim == 2:
        if value.shape[-1] != sample_dim:
            raise ValueError(f"{name} last dimension must be {sample_dim}, got {value.shape}")
        return value[:, None, :]
    if value.ndim == 3:
        if value.shape[-1] != sample_dim:
            raise ValueError(f"{name} last dimension must be {sample_dim}, got {value.shape}")
        return value
    if value.ndim == 4:
        value = value.reshape(value.shape[0], value.shape[1], -1)
        if value.shape[-1] != sample_dim:
            raise ValueError(f"{name} must flatten to sample dim {sample_dim}, got {value.shape}")
        return value
    raise ValueError(f"{name} must have rank 2, 3, or 4, got {value.shape}")


def _as_gen_field(value, ref, batch_size, count, sample_dim, name):
    if value is None:
        return jnp.zeros_like(ref)
    value = jnp.asarray(value, dtype=ref.dtype)
    if value.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch dimension {value.shape[0]} does not match gen batch {batch_size}"
        )
    if value.ndim == 2:
        if count != 1 or value.shape[-1] != sample_dim:
            raise ValueError(f"{name} with rank 2 only works for one gen particle")
        return value[:, None, :]
    if value.ndim == 3:
        if value.shape[1] != count or value.shape[-1] != sample_dim:
            raise ValueError(
                f"{name} must have shape [B, {count}, {sample_dim}], got {value.shape}"
            )
        return value
    if value.ndim == 4:
        value = value.reshape(value.shape[0], value.shape[1], -1)
        if value.shape[1] != count or value.shape[-1] != sample_dim:
            raise ValueError(
                f"{name} must flatten to [B, {count}, {sample_dim}], got {value.shape}"
            )
        return value
    raise ValueError(f"{name} must have rank 2, 3, or 4, got {value.shape}")


def _as_weight(value, ref, batch_size, count, name, default):
    if value is None:
        return jnp.full((batch_size, count), default, dtype=ref.dtype)
    value = jnp.asarray(value, dtype=ref.dtype)
    if value.ndim == 0:
        return jnp.full((batch_size, count), value, dtype=ref.dtype)
    if value.ndim == 1:
        if value.shape[0] == batch_size:
            return jnp.broadcast_to(value[:, None], (batch_size, count))
        if value.shape[0] == count:
            return jnp.broadcast_to(value[None, :], (batch_size, count))
    if value.ndim == 2 and value.shape == (batch_size, count):
        return value
    raise ValueError(f"{name} must be broadcastable to [{batch_size}, {count}], got {value.shape}")


def compute_positive_drift_velocity(
    gen_actions,
    positive_actions,
    positive_weights,
    kernel_type,
    kernel_bandwidth,
    negative_actions=None,
    lambda_pi=1.0,
    beta=1.0,
    positive_drift_type="score",
    exclude_self_kde=True,
    adaptive_bandwidth=False,
    bandwidth_quantile=0.5,
    bandwidth_scale=1.0,
    min_bandwidth=1e-3,
    max_bandwidth=None,
    compact_kernel_eps=1e-6,
    compact_kernel_score_clip=100.0,
    compact_kernel_min_neighbors=1,
    compact_kernel_fallback="nearest",
    compact_fallback_bandwidth=None,
):
    """Compute value-weighted positive drifting velocity.

    Q is not evaluated here.  It only affects this field through the detached
    ``positive_weights`` supplied by the caller.
    """
    gen = jnp.asarray(gen_actions, dtype=jnp.float32)
    positive_actions = jnp.asarray(positive_actions, dtype=gen.dtype)
    positive_weights = jnp.asarray(positive_weights, dtype=gen.dtype)
    negative_actions = gen if negative_actions is None else jnp.asarray(negative_actions, dtype=gen.dtype)

    if gen.ndim != 3 or positive_actions.ndim != 3 or negative_actions.ndim != 3:
        raise ValueError("gen_actions, positive_actions, and negative_actions must be rank-3 [B, N, S]")

    kernel_bandwidth = float(kernel_bandwidth)
    compact_fallback_bandwidth_value = (
        None
        if compact_fallback_bandwidth is None or float(compact_fallback_bandwidth) <= 0.0
        else float(compact_fallback_bandwidth)
    )
    positive_bandwidth = _resolve_bandwidth(
        gen,
        positive_actions,
        kernel_bandwidth,
        adaptive=bool(adaptive_bandwidth),
        quantile=float(bandwidth_quantile),
        scale=float(bandwidth_scale),
        min_bandwidth=float(min_bandwidth),
        max_bandwidth=max_bandwidth,
        exclude_self=False,
    )
    negative_bandwidth = _resolve_bandwidth(
        gen,
        negative_actions,
        kernel_bandwidth,
        adaptive=bool(adaptive_bandwidth),
        quantile=float(bandwidth_quantile),
        scale=float(bandwidth_scale),
        min_bandwidth=float(min_bandwidth),
        max_bandwidth=max_bandwidth,
        exclude_self=bool(exclude_self_kde),
    )

    drift_type = str(positive_drift_type)
    if drift_type not in ("score", "positive_drift_score", "meanshift"):
        raise ValueError(f"Unsupported positive_drift_type: {positive_drift_type}")

    positive_score, compact_info = _kde_score(
        gen,
        positive_actions,
        positive_bandwidth,
        fallback_bandwidth=compact_fallback_bandwidth_value,
        ref_weights=positive_weights,
        kernel=str(kernel_type),
        compact_eps=float(compact_kernel_eps),
        compact_score_clip=float(compact_kernel_score_clip),
        compact_min_neighbors=int(compact_kernel_min_neighbors),
        compact_fallback=str(compact_kernel_fallback),
        return_info=True,
    )
    negative_score = _kde_score(
        gen,
        negative_actions,
        negative_bandwidth,
        fallback_bandwidth=compact_fallback_bandwidth_value,
        exclude_self=bool(exclude_self_kde),
        kernel=str(kernel_type),
        compact_eps=float(compact_kernel_eps),
        compact_score_clip=float(compact_kernel_score_clip),
        compact_min_neighbors=int(compact_kernel_min_neighbors),
        compact_fallback=str(compact_kernel_fallback),
    )

    if drift_type == "meanshift":
        h2 = jnp.square(_format_bandwidth(kernel_bandwidth, gen, positive_actions))
        positive_score = positive_score * h2
        negative_score = negative_score * h2

    positive_velocity = float(beta) * positive_score
    negative_velocity = -float(lambda_pi) * negative_score
    velocity = positive_velocity + negative_velocity
    positive_norm = jnp.linalg.norm(positive_score.reshape((-1, positive_score.shape[-1])), axis=-1)
    negative_norm = jnp.linalg.norm(negative_score.reshape((-1, negative_score.shape[-1])), axis=-1)
    velocity_norm = jnp.linalg.norm(velocity.reshape((-1, velocity.shape[-1])), axis=-1)
    info = {
        "positive_score_rms": _rms(positive_score),
        "negative_score_rms": _rms(negative_score),
        "positive_velocity_rms": _rms(positive_velocity),
        "negative_velocity_rms": _rms(negative_velocity),
        "total_velocity_rms": _rms(velocity),
        "cos_positive_negative": _safe_cos(positive_score, negative_score),
        "cos_positive_total": _safe_cos(positive_score, velocity),
        "max_positive_score_norm": jnp.max(positive_norm),
        "p99_positive_score_norm": _norm_quantile(positive_score, 0.99),
        "max_negative_score_norm": jnp.max(negative_norm),
        "p99_negative_score_norm": _norm_quantile(negative_score, 0.99),
        "max_total_velocity_norm": jnp.max(velocity_norm),
        "p99_total_velocity_norm": _norm_quantile(velocity, 0.99),
        "positive_bandwidth_mean": jnp.mean(positive_bandwidth),
        "negative_bandwidth_mean": jnp.mean(negative_bandwidth),
        **compact_info,
    }
    debug = {
        "positive_score": positive_score,
        "negative_score": negative_score,
        "positive_velocity": positive_velocity,
        "negative_velocity": negative_velocity,
        "total_velocity": velocity,
    }
    return velocity, positive_score, negative_score, info, debug


def offline_transport_drift_loss(
    gen,
    offline_actions=None,
    offline_weights=None,
    positive_actions=None,
    positive_weights=None,
    adv_grad=None,
    fixed_pos=None,
    tau=1.0,
    beta=1.0,
    lambda_pi=1.0,
    kernel_bandwidth=0.25,
    transport_step_size=0.05,
    normalize_q_grad=False,
    q_grad_clip_norm=None,
    exclude_self_kde=True,
    adaptive_bandwidth=False,
    bandwidth_quantile=0.5,
    bandwidth_scale=1.0,
    min_bandwidth=1e-3,
    max_bandwidth=None,
    kde_kernel="gaussian",
    compact_kernel_eps=1e-6,
    compact_kernel_score_clip=100.0,
    compact_kernel_min_neighbors=1,
    compact_kernel_fallback="nearest",
    compact_fallback_bandwidth=None,
    actor_update_mode="decomposed",
    positive_drift_type="score",
    override_behavior_score=None,
    override_adv_grad=None,
    return_debug=False,
):
    """Score-difference offline transport loss.

    Args:
        gen: Generated actor particles with shape [B, G, S].
        offline_actions: Static behavior support particles with shape [B, K, S].
        offline_weights: Optional non-negative behavior support weights [B, K].
        positive_actions: Optional positive target particles [B, K, S] for positive_drift mode.
        positive_weights: Optional detached positive target weights [B, K] for positive_drift mode.
        adv_grad: Optional detached critic action-gradient field [B, G, S].

    Returns:
        Per-example loss [B] and scalar diagnostics.
    """
    if gen.ndim != 3:
        raise ValueError(f"gen must have shape [B, G, S], got {gen.shape}")

    if offline_actions is None:
        offline_actions = fixed_pos
    if offline_actions is None:
        raise ValueError("offline_transport_drift_loss requires offline_actions")

    gen = gen.astype(jnp.float32)
    batch_size, gen_count, sample_dim = gen.shape
    offline_actions = _as_particle_set(
        offline_actions,
        gen,
        batch_size,
        sample_dim,
        "offline_actions",
    )
    if offline_actions.shape[1] <= 0:
        raise ValueError("offline_transport_drift_loss requires at least one offline action")
    offline_weights = _as_weight(
        offline_weights,
        gen,
        batch_size,
        offline_actions.shape[1],
        "offline_weights",
        1.0,
    )
    if override_adv_grad is not None:
        adv_grad = override_adv_grad
    adv_grad = _as_gen_field(adv_grad, gen, batch_size, gen_count, sample_dim, "adv_grad")

    tau = max(float(tau), 1e-8)
    beta = float(beta)
    lambda_pi = float(lambda_pi)
    kernel_bandwidth = float(kernel_bandwidth)
    compact_fallback_bandwidth_value = (
        None
        if compact_fallback_bandwidth is None or float(compact_fallback_bandwidth) <= 0.0
        else float(compact_fallback_bandwidth)
    )
    kde_kernel = str(kde_kernel)
    transport_step_size = jnp.asarray(transport_step_size, dtype=gen.dtype)
    actor_update_mode = str(actor_update_mode)
    if actor_update_mode not in ("decomposed", "positive_drift", "positive_drift_plus_residual"):
        raise ValueError(f"Unsupported actor_update_mode: {actor_update_mode}")

    old_gen = jax.lax.stop_gradient(gen)
    offline_actions = jax.lax.stop_gradient(offline_actions)
    offline_weights = jax.lax.stop_gradient(offline_weights)
    q_grad = jax.lax.stop_gradient(adv_grad)

    if q_grad_clip_norm is not None and q_grad_clip_norm > 0:
        q_norm = jnp.clip(jnp.linalg.norm(q_grad, axis=-1, keepdims=True), min=1e-8)
        clip_scale = jnp.minimum(float(q_grad_clip_norm) / q_norm, 1.0)
        q_grad = q_grad * clip_scale

    q_grad_rms_raw = jnp.sqrt(jnp.clip(jnp.mean(jnp.square(q_grad)), min=1e-12))
    if normalize_q_grad:
        q_grad = q_grad / q_grad_rms_raw

    q_score = q_grad / tau
    behavior_bandwidth = _resolve_bandwidth(
        old_gen,
        offline_actions,
        kernel_bandwidth,
        adaptive=bool(adaptive_bandwidth),
        quantile=float(bandwidth_quantile),
        scale=float(bandwidth_scale),
        min_bandwidth=float(min_bandwidth),
        max_bandwidth=max_bandwidth,
        exclude_self=False,
    )
    policy_bandwidth = _resolve_bandwidth(
        old_gen,
        old_gen,
        kernel_bandwidth,
        adaptive=bool(adaptive_bandwidth),
        quantile=float(bandwidth_quantile),
        scale=float(bandwidth_scale),
        min_bandwidth=float(min_bandwidth),
        max_bandwidth=max_bandwidth,
        exclude_self=bool(exclude_self_kde),
    )
    if override_behavior_score is None:
        behavior_score, compact_info = _kde_score(
            old_gen,
            offline_actions,
            behavior_bandwidth,
            fallback_bandwidth=compact_fallback_bandwidth_value,
            ref_weights=offline_weights,
            kernel=kde_kernel,
            compact_eps=float(compact_kernel_eps),
            compact_score_clip=float(compact_kernel_score_clip),
            compact_min_neighbors=int(compact_kernel_min_neighbors),
            compact_fallback=str(compact_kernel_fallback),
            return_info=True,
        )
    else:
        behavior_score = _as_gen_field(
            override_behavior_score,
            old_gen,
            batch_size,
            gen_count,
            sample_dim,
            "override_behavior_score",
        )
        behavior_score = jax.lax.stop_gradient(behavior_score)
        compact_info = {
            "compact_no_neighbor_fraction": jnp.asarray(0.0, dtype=gen.dtype),
            "compact_inside_neighbor_count_mean": jnp.asarray(0.0, dtype=gen.dtype),
            "compact_inside_neighbor_count_min": jnp.asarray(0.0, dtype=gen.dtype),
            "compact_density_mean": jnp.asarray(0.0, dtype=gen.dtype),
            "compact_density_min": jnp.asarray(0.0, dtype=gen.dtype),
        }
    policy_score = _kde_score(
        old_gen,
        old_gen,
        policy_bandwidth,
        fallback_bandwidth=compact_fallback_bandwidth_value,
        exclude_self=bool(exclude_self_kde),
        kernel=kde_kernel,
        compact_eps=float(compact_kernel_eps),
        compact_score_clip=float(compact_kernel_score_clip),
        compact_min_neighbors=int(compact_kernel_min_neighbors),
        compact_fallback=str(compact_kernel_fallback),
    )
    positive_score = behavior_score
    negative_score = policy_score
    positive_velocity = beta * positive_score
    negative_velocity = -lambda_pi * negative_score
    positive_debug = {}
    if actor_update_mode in ("positive_drift", "positive_drift_plus_residual"):
        if positive_actions is None:
            positive_actions = offline_actions
        if positive_weights is None:
            positive_weights = offline_weights
        positive_actions = _as_particle_set(
            positive_actions,
            gen,
            batch_size,
            sample_dim,
            "positive_actions",
        )
        positive_weights = _as_weight(
            positive_weights,
            gen,
            batch_size,
            positive_actions.shape[1],
            "positive_weights",
            1.0,
        )
        positive_actions = jax.lax.stop_gradient(positive_actions)
        positive_weights = jax.lax.stop_gradient(positive_weights)
        q_score = jnp.zeros_like(q_score)
        velocity, positive_score, negative_score, positive_info, positive_debug = (
            compute_positive_drift_velocity(
                old_gen,
                positive_actions,
                positive_weights,
                kde_kernel,
                kernel_bandwidth,
                negative_actions=old_gen,
                lambda_pi=lambda_pi,
                beta=beta,
                positive_drift_type=positive_drift_type,
                exclude_self_kde=bool(exclude_self_kde),
                adaptive_bandwidth=bool(adaptive_bandwidth),
                bandwidth_quantile=float(bandwidth_quantile),
                bandwidth_scale=float(bandwidth_scale),
                min_bandwidth=float(min_bandwidth),
                max_bandwidth=max_bandwidth,
                compact_kernel_eps=float(compact_kernel_eps),
                compact_kernel_score_clip=float(compact_kernel_score_clip),
                compact_kernel_min_neighbors=int(compact_kernel_min_neighbors),
                compact_kernel_fallback=str(compact_kernel_fallback),
                compact_fallback_bandwidth=compact_fallback_bandwidth_value,
            )
        )
        behavior_score = positive_score
        policy_score = negative_score
        positive_velocity = positive_debug["positive_velocity"]
        negative_velocity = positive_debug["negative_velocity"]
        compact_info = {
            "compact_no_neighbor_fraction": positive_info["compact_no_neighbor_fraction"],
            "compact_inside_neighbor_count_mean": positive_info["compact_inside_neighbor_count_mean"],
            "compact_inside_neighbor_count_min": positive_info["compact_inside_neighbor_count_min"],
            "compact_density_mean": positive_info["compact_density_mean"],
            "compact_density_min": positive_info["compact_density_min"],
        }
    else:
        velocity = q_score + positive_velocity + negative_velocity
    gen_target = old_gen + transport_step_size * velocity

    target_step = gen_target - old_gen
    diff = gen - jax.lax.stop_gradient(gen_target)
    loss = jnp.mean(jnp.square(diff), axis=(-1, -2))
    behavior_norm = jnp.linalg.norm(behavior_score.reshape((-1, sample_dim)), axis=-1)
    positive_norm = jnp.linalg.norm(positive_score.reshape((-1, sample_dim)), axis=-1)
    negative_norm = jnp.linalg.norm(negative_score.reshape((-1, sample_dim)), axis=-1)
    velocity_norm = jnp.linalg.norm(velocity.reshape((-1, sample_dim)), axis=-1)
    nonfinite_seen = (
        jnp.any(~jnp.isfinite(loss))
        | jnp.any(~jnp.isfinite(q_score))
        | jnp.any(~jnp.isfinite(behavior_score))
        | jnp.any(~jnp.isfinite(policy_score))
        | jnp.any(~jnp.isfinite(velocity))
        | jnp.any(~jnp.isfinite(gen_target))
    )
    info = {
        "transport_loss_type": jnp.asarray(1.0, dtype=gen.dtype),
        "actor_update_mode_is_positive": jnp.asarray(
            float(actor_update_mode in ("positive_drift", "positive_drift_plus_residual")),
            dtype=gen.dtype,
        ),
        "tau": jnp.asarray(tau, dtype=gen.dtype),
        "beta": jnp.asarray(beta, dtype=gen.dtype),
        "lambda_pi": jnp.asarray(lambda_pi, dtype=gen.dtype),
        "kernel_bandwidth": jnp.asarray(kernel_bandwidth, dtype=gen.dtype),
        "compact_fallback_bandwidth": jnp.asarray(
            kernel_bandwidth
            if compact_fallback_bandwidth_value is None
            else compact_fallback_bandwidth_value,
            dtype=gen.dtype,
        ),
        "kde_kernel_is_compact": jnp.asarray(float(kde_kernel != "gaussian"), dtype=gen.dtype),
        "transport_step_size": jnp.asarray(transport_step_size, dtype=gen.dtype),
        "normalize_q_grad": jnp.asarray(float(bool(normalize_q_grad)), dtype=gen.dtype),
        "q_grad_clip_norm": jnp.asarray(
            0.0 if q_grad_clip_norm is None else float(q_grad_clip_norm),
            dtype=gen.dtype,
        ),
        "exclude_self_kde": jnp.asarray(float(bool(exclude_self_kde)), dtype=gen.dtype),
        "adaptive_bandwidth": jnp.asarray(float(bool(adaptive_bandwidth)), dtype=gen.dtype),
        "behavior_bandwidth_mean": jnp.mean(behavior_bandwidth),
        "policy_bandwidth_mean": jnp.mean(policy_bandwidth),
        "q_grad_rms_raw": q_grad_rms_raw,
        "q_score_rms": _rms(q_score),
        "behavior_score_rms": _rms(behavior_score),
        "policy_score_rms": _rms(policy_score),
        "positive_score_rms": _rms(positive_score),
        "negative_score_rms": _rms(negative_score),
        "positive_velocity_rms": _rms(positive_velocity),
        "negative_velocity_rms": _rms(negative_velocity),
        "velocity_rms": _rms(velocity),
        "total_velocity_rms": _rms(velocity),
        "target_step_rms": _rms(target_step),
        "cos_positive_negative": _safe_cos(positive_score, negative_score),
        "cos_positive_total": _safe_cos(positive_score, velocity),
        "max_behavior_score_norm": jnp.max(behavior_norm),
        "p99_behavior_score_norm": _norm_quantile(behavior_score, 0.99),
        "max_positive_score_norm": jnp.max(positive_norm),
        "p99_positive_score_norm": _norm_quantile(positive_score, 0.99),
        "max_negative_score_norm": jnp.max(negative_norm),
        "p99_negative_score_norm": _norm_quantile(negative_score, 0.99),
        "max_total_velocity_norm": jnp.max(velocity_norm),
        "p99_total_velocity_norm": _norm_quantile(velocity, 0.99),
        "nan_or_inf_seen": jnp.asarray(nonfinite_seen, dtype=gen.dtype),
        "n_offline": jnp.asarray(float(offline_actions.shape[1]), dtype=gen.dtype),
        "offline_action_count": jnp.asarray(float(offline_actions.shape[1]), dtype=gen.dtype),
        **compact_info,
    }
    if not return_debug:
        return loss, info
    debug = {
        "gen": old_gen,
        "target": gen_target,
        "q_score": q_score,
        "behavior_score": beta * behavior_score,
        "policy_score": -lambda_pi * policy_score,
        "raw_policy_score": policy_score,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "positive_velocity": positive_velocity,
        "negative_velocity": negative_velocity,
        "total_velocity": velocity,
        "target_step": target_step,
        "offline_actions": offline_actions,
        "offline_weights": offline_weights,
        **positive_debug,
    }
    return loss, info, debug
