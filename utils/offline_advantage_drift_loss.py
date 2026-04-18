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


def _kde_score(query, refs, bandwidth, ref_weights=None, exclude_self=False, eps=1e-8):
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


def offline_transport_drift_loss(
    gen,
    offline_actions=None,
    offline_weights=None,
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
):
    """Score-difference offline transport loss.

    Args:
        gen: Generated actor particles with shape [B, G, S].
        offline_actions: Static behavior support particles with shape [B, K, S].
        offline_weights: Optional non-negative behavior support weights [B, K].
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
    adv_grad = _as_gen_field(adv_grad, gen, batch_size, gen_count, sample_dim, "adv_grad")

    tau = max(float(tau), 1e-8)
    beta = float(beta)
    lambda_pi = float(lambda_pi)
    kernel_bandwidth = float(kernel_bandwidth)
    transport_step_size = float(transport_step_size)

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
    behavior_score = _kde_score(
        old_gen,
        offline_actions,
        behavior_bandwidth,
        ref_weights=offline_weights,
    )
    policy_score = _kde_score(
        old_gen,
        old_gen,
        policy_bandwidth,
        exclude_self=bool(exclude_self_kde),
    )
    velocity = q_score + beta * behavior_score - lambda_pi * policy_score
    gen_target = old_gen + transport_step_size * velocity

    target_step = gen_target - old_gen
    diff = gen - jax.lax.stop_gradient(gen_target)
    loss = jnp.mean(jnp.square(diff), axis=(-1, -2))
    info = {
        "transport_loss_type": jnp.asarray(1.0, dtype=gen.dtype),
        "tau": jnp.asarray(tau, dtype=gen.dtype),
        "beta": jnp.asarray(beta, dtype=gen.dtype),
        "lambda_pi": jnp.asarray(lambda_pi, dtype=gen.dtype),
        "kernel_bandwidth": jnp.asarray(kernel_bandwidth, dtype=gen.dtype),
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
        "q_score_rms": jnp.sqrt(jnp.clip(jnp.mean(jnp.square(q_score)), min=1e-12)),
        "behavior_score_rms": jnp.sqrt(
            jnp.clip(jnp.mean(jnp.square(behavior_score)), min=1e-12)
        ),
        "policy_score_rms": jnp.sqrt(
            jnp.clip(jnp.mean(jnp.square(policy_score)), min=1e-12)
        ),
        "velocity_rms": jnp.sqrt(jnp.clip(jnp.mean(jnp.square(velocity)), min=1e-12)),
        "target_step_rms": jnp.sqrt(jnp.clip(jnp.mean(jnp.square(target_step)), min=1e-12)),
        "n_offline": jnp.asarray(float(offline_actions.shape[1]), dtype=gen.dtype),
        "offline_action_count": jnp.asarray(float(offline_actions.shape[1]), dtype=gen.dtype),
    }
    return loss, info
