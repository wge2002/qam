import jax
import jax.numpy as jnp

from agents.drift import DriftAgent, _normalize_weights, aggregate_q_values, get_config as get_drift_config


def _cdist(x, y, eps=1.0e-8):
    xydot = jnp.einsum("bnd,bmd->bnm", x, y)
    xnorms = jnp.einsum("bnd,bnd->bn", x, x)
    ynorms = jnp.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2.0 * xydot
    return jnp.sqrt(jnp.clip(sq_dist, a_min=eps))


def _rms(value):
    return jnp.sqrt(jnp.maximum(jnp.mean(jnp.square(value)), 0.0))


def _norm_quantile(value, quantile):
    norm = jnp.linalg.norm(jnp.reshape(value, (-1, value.shape[-1])), axis=-1)
    norm = jnp.sort(norm)
    idx = int(round(float(quantile) * (norm.shape[0] - 1)))
    return norm[idx]


def affinity_q_goal_loss(
    gen,
    fixed_pos,
    q_grad,
    fixed_neg=None,
    weight_gen=None,
    weight_pos=None,
    weight_neg=None,
    r_list=(0.02, 0.05, 0.2),
    goal_step_scale=1.0,
    goal_clip_norm=0.0,
    q_step_size=0.05,
    q_grad_scale=1.0,
    q_grad_clip_norm=None,
    normalize_q_grad=False,
    q_tau=0.75,
    total_goal_clip_norm=0.0,
):
    batch_size, num_gen, sample_dim = gen.shape
    num_pos = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = jnp.zeros((batch_size, 0, sample_dim), dtype=gen.dtype)
    num_neg = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = jnp.ones((batch_size, num_gen), dtype=gen.dtype)
    if weight_pos is None:
        weight_pos = jnp.ones((batch_size, num_pos), dtype=gen.dtype)
    if weight_neg is None:
        weight_neg = jnp.ones((batch_size, num_neg), dtype=gen.dtype)

    gen = gen.astype(jnp.float32)
    fixed_pos = jax.lax.stop_gradient(fixed_pos.astype(jnp.float32))
    fixed_neg = jax.lax.stop_gradient(fixed_neg.astype(jnp.float32))
    weight_gen = jax.lax.stop_gradient(weight_gen.astype(jnp.float32))
    weight_pos = jax.lax.stop_gradient(weight_pos.astype(jnp.float32))
    weight_neg = jax.lax.stop_gradient(weight_neg.astype(jnp.float32))
    old_gen = jax.lax.stop_gradient(gen)

    targets = jnp.concatenate([old_gen, fixed_neg, fixed_pos], axis=1)
    targets_w = jnp.concatenate([weight_gen, weight_neg, weight_pos], axis=1)

    dist = _cdist(old_gen, targets)
    weighted_dist = dist * targets_w[:, None, :]
    scale = weighted_dist.mean() / jnp.clip(targets_w.mean(), min=1.0e-8)
    scale_inputs = jnp.clip(scale / jnp.sqrt(float(sample_dim)), min=1.0e-3)
    old_gen_scaled = old_gen / scale_inputs
    targets_scaled = targets / scale_inputs

    dist_normed = dist / jnp.clip(scale, min=1.0e-3)
    diag_mask = jnp.eye(num_gen, dtype=gen.dtype)
    block_mask = jnp.concatenate(
        [diag_mask, jnp.zeros((num_gen, num_neg + num_pos), dtype=gen.dtype)],
        axis=1,
    )[None, :, :]
    dist_normed = dist_normed + block_mask * 100.0

    affinity_force = jnp.zeros_like(old_gen_scaled)
    info = {"drift_scale": scale}
    for r in tuple(r_list):
        logits = -dist_normed / float(r)
        affinity = jax.nn.softmax(logits, axis=-1)
        affinity_t = jax.nn.softmax(logits, axis=-2)
        affinity = jnp.sqrt(jnp.clip(affinity * affinity_t, min=1.0e-6))
        affinity = affinity * targets_w[:, None, :]

        split_idx = num_gen + num_neg
        aff_neg = affinity[:, :, :split_idx]
        aff_pos = affinity[:, :, split_idx:]

        sum_pos = aff_pos.sum(axis=-1, keepdims=True)
        r_coeff_neg = -aff_neg * sum_pos
        sum_neg = aff_neg.sum(axis=-1, keepdims=True)
        r_coeff_pos = aff_pos * sum_neg
        r_coeff = jnp.concatenate([r_coeff_neg, r_coeff_pos], axis=2)

        total_force_r = jnp.einsum("biy,byx->bix", r_coeff, targets_scaled)
        total_coeffs = r_coeff.sum(axis=-1)
        total_force_r = total_force_r - total_coeffs[:, :, None] * old_gen_scaled

        force_norm = jnp.mean(jnp.square(total_force_r))
        info[f"drift_loss_{r}"] = force_norm
        affinity_force = affinity_force + total_force_r / jnp.sqrt(jnp.clip(force_norm, min=1.0e-8))

    affinity_force = affinity_force * float(goal_step_scale)
    if float(goal_clip_norm) > 0.0:
        force_norm = jnp.linalg.norm(affinity_force, axis=-1, keepdims=True)
        affinity_force = affinity_force * jnp.minimum(
            1.0,
            float(goal_clip_norm) / jnp.clip(force_norm, min=1.0e-8),
        )

    q_grad = jax.lax.stop_gradient(q_grad.astype(jnp.float32)) * float(q_grad_scale)
    if q_grad_clip_norm is not None and float(q_grad_clip_norm) > 0.0:
        q_norm = jnp.clip(jnp.linalg.norm(q_grad, axis=-1, keepdims=True), min=1.0e-8)
        q_grad = q_grad * jnp.minimum(float(q_grad_clip_norm) / q_norm, 1.0)
    q_grad_rms_raw = _rms(q_grad)
    if bool(normalize_q_grad):
        q_grad = q_grad / jnp.clip(q_grad_rms_raw, min=1.0e-8)

    affinity_goal_scaled = old_gen_scaled + affinity_force
    affinity_goal = affinity_goal_scaled * scale_inputs
    q_score = q_grad / jnp.clip(float(q_tau), min=1.0e-8)
    q_step = jnp.asarray(q_step_size, dtype=gen.dtype) * q_score
    goal = affinity_goal + q_step
    if float(total_goal_clip_norm) > 0.0:
        goal_delta = goal - old_gen
        goal_delta_norm = jnp.linalg.norm(goal_delta, axis=-1, keepdims=True)
        goal_delta = goal_delta * jnp.minimum(
            1.0,
            float(total_goal_clip_norm) / jnp.clip(goal_delta_norm, min=1.0e-8),
        )
        goal = old_gen + goal_delta
    goal_scaled = goal / jnp.clip(scale_inputs, min=1.0e-8)
    diff = gen / jax.lax.stop_gradient(scale_inputs) - jax.lax.stop_gradient(goal_scaled)
    loss = jnp.mean(jnp.square(diff), axis=(-1, -2))

    affinity_delta = affinity_goal - old_gen
    goal_delta = goal - old_gen
    info.update(
        {
            "target_step_rms": _rms(goal_delta),
            "affinity_step_rms": _rms(affinity_delta),
            "q_step_rms": _rms(q_step),
            "q_grad_rms_raw": q_grad_rms_raw,
            "q_score_rms": _rms(q_score),
            "total_velocity_rms": _rms(goal_delta),
            "max_total_velocity_norm": jnp.max(jnp.linalg.norm(goal_delta, axis=-1)),
            "p99_total_velocity_norm": _norm_quantile(goal_delta, 0.99),
        }
    )
    return loss, {key: jnp.mean(value) for key, value in info.items()}


class DriftAffinityQAgent(DriftAgent):
    """Drift critic with drifting-policy affinity goal replacing KDE behavior/self terms."""

    def _affinity_actor_loss(self, gen_actions, support_actions, support_weights, q_grad):
        if bool(self.config.get("affinity_per_timestep_loss", True)):
            horizon = int(self.config["horizon_length"]) if bool(self.config["action_chunking"]) else 1
            step_dim = gen_actions.shape[-1] // horizon
            gen_steps = jnp.reshape(gen_actions, gen_actions.shape[:-1] + (horizon, step_dim))
            support_steps = jnp.reshape(support_actions, support_actions.shape[:-1] + (horizon, step_dim))
            q_grad_steps = jnp.reshape(q_grad, q_grad.shape[:-1] + (horizon, step_dim))
            total_loss = 0.0
            total_info = None
            for t in range(horizon):
                loss_t, info_t = affinity_q_goal_loss(
                    gen_steps[:, :, t, :],
                    support_steps[:, :, t, :],
                    q_grad_steps[:, :, t, :],
                    weight_pos=support_weights,
                    r_list=self.config["affinity_temperatures"],
                    goal_step_scale=self.config["affinity_goal_step_scale"],
                    goal_clip_norm=self.config["affinity_goal_clip_norm"],
                    q_step_size=self._transport_step_size(),
                    q_grad_scale=self.config["q_grad_scale"],
                    q_grad_clip_norm=self.config["q_grad_clip_norm"],
                    normalize_q_grad=self.config["normalize_q_grad"],
                    q_tau=self.config["drift_tau"],
                    total_goal_clip_norm=self.config["total_goal_clip_norm"],
                )
                total_loss = total_loss + loss_t
                if total_info is None:
                    total_info = info_t
                else:
                    total_info = {key: total_info[key] + info_t[key] for key in total_info}
            return total_loss / horizon, {key: value / horizon for key, value in total_info.items()}

        return affinity_q_goal_loss(
            gen_actions,
            support_actions,
            q_grad,
            weight_pos=support_weights,
            r_list=self.config["affinity_temperatures"],
            goal_step_scale=self.config["affinity_goal_step_scale"],
            goal_clip_norm=self.config["affinity_goal_clip_norm"],
            q_step_size=self._transport_step_size(),
            q_grad_scale=self.config["q_grad_scale"],
            q_grad_clip_norm=self.config["q_grad_clip_norm"],
            normalize_q_grad=self.config["normalize_q_grad"],
            q_tau=self.config["drift_tau"],
            total_goal_clip_norm=self.config["total_goal_clip_norm"],
        )

    def actor_loss(self, batch, grad_params, rng):
        batch_actions = self._batch_actions(batch)
        batch_size, action_dim = batch_actions.shape
        support_actions, support_weights = self._support_actions(batch, batch_actions)
        support_weights = jax.lax.stop_gradient(support_weights)

        rng, gen_rng = jax.random.split(rng)
        gen_per_obs = int(self.config["gen_per_obs"])
        noises = jax.random.normal(gen_rng, (batch_size, gen_per_obs, self.config["noise_dim"]))
        obs_rep = jnp.repeat(batch["observations"][:, None, :], gen_per_obs, axis=1)
        gen_actions = self.network.select("actor")(obs_rep, noises, params=grad_params)
        gen_actions = jnp.clip(gen_actions, -1.0, 1.0)

        flat_obs = obs_rep.reshape(batch_size * gen_per_obs, batch["observations"].shape[-1])
        flat_actions = jax.lax.stop_gradient(gen_actions).reshape(batch_size * gen_per_obs, action_dim)

        def q_sum(actions):
            q_values = self.network.select("target_critic")(flat_obs, actions)
            q = aggregate_q_values(
                q_values,
                self.config["actor_q_agg"],
                pessimism_coef=self.config["actor_pessimism_coef"],
            )
            return q.sum()

        q_grad = jax.grad(q_sum)(flat_actions).reshape(batch_size, gen_per_obs, action_dim)
        gen_q_values = self.network.select("target_critic")(flat_obs, flat_actions)
        gen_q = aggregate_q_values(
            gen_q_values,
            self.config["actor_q_agg"],
            pessimism_coef=self.config["actor_pessimism_coef"],
        ).reshape(batch_size, gen_per_obs)

        support_q = self._score_action_set(
            batch["observations"],
            support_actions,
            self.config["actor_q_agg"],
            pessimism_coef=self.config["actor_pessimism_coef"],
            critic="target_critic",
        )
        data_q = jnp.sum(support_q * _normalize_weights(support_weights), axis=1)

        loss, info = self._affinity_actor_loss(
            gen_actions,
            support_actions,
            support_weights,
            q_grad,
        )
        actor_loss = loss.mean()
        support_mse = jnp.min(
            jnp.mean(jnp.square(gen_actions[:, 0, None, :] - support_actions), axis=-1),
            axis=-1,
        ).mean()

        info = {
            **info,
            "actor_loss": actor_loss,
            "positive_score_rms": info["affinity_step_rms"],
            "behavior_score_rms": info["affinity_step_rms"],
            "negative_score_rms": jnp.asarray(0.0, dtype=actor_loss.dtype),
            "policy_score_rms": jnp.asarray(0.0, dtype=actor_loss.dtype),
            "gen_q_mean": gen_q.mean(),
            "data_q_mean": data_q.mean(),
            "data_advantage_mean": (data_q - gen_q.mean(axis=1)).mean(),
            "support_q_std_mean": support_q.std(axis=1).mean(),
            "support_mse": support_mse,
            "transport_step_size": self._transport_step_size(),
        }
        return actor_loss, info


def get_config():
    config = get_drift_config()
    config.agent_name = "drift_affinity_q"
    config.affinity_temperatures = (0.02, 0.05, 0.2)
    config.affinity_per_timestep_loss = True
    config.affinity_goal_step_scale = 1.0
    config.affinity_goal_clip_norm = 0.0
    config.total_goal_clip_norm = 0.0
    return config
