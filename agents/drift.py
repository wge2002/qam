import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, Value
from utils.offline_advantage_drift_loss import offline_transport_drift_loss


class TanhImplicitActor(nn.Module):
    """Implicit actor used by offline transport drift."""

    action_dim: int
    hidden_dims: tuple
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations, noises):
        x = jnp.concatenate([observations, noises], axis=-1)
        x = MLP(
            (*self.hidden_dims, self.action_dim),
            activations=nn.gelu,
            activate_final=False,
            layer_norm=self.layer_norm,
        )(x)
        return jnp.tanh(x)


def aggregate_q_values(q_values, mode, pessimism_coef=0.0):
    if q_values.ndim == 1:
        return q_values
    if mode == "min":
        return jnp.min(q_values, axis=0)
    if mode == "max":
        return jnp.max(q_values, axis=0)
    if mode == "mean":
        return jnp.mean(q_values, axis=0)
    if mode in ("pessimistic", "mean_minus_std", "mean-std"):
        return jnp.mean(q_values, axis=0) - float(pessimism_coef) * jnp.std(q_values, axis=0)
    raise ValueError(f"Unsupported q_agg: {mode}")


def aggregate_sample_values(values, mode):
    if mode == "mean":
        return jnp.mean(values, axis=1)
    if mode == "min":
        return jnp.min(values, axis=1)
    if mode == "max":
        return jnp.max(values, axis=1)
    raise ValueError(f"Unsupported sample value aggregation: {mode}")


def _normalize_weights(weights, eps=1e-8):
    weights = jnp.clip(weights, min=0.0)
    return weights / jnp.clip(jnp.sum(weights, axis=-1, keepdims=True), min=eps)


class DriftAgent(flax.struct.PyTreeNode):
    """JAX/Flax port of the offline transport drift actor."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _batch_actions(self, batch):
        if self.config["action_chunking"]:
            return jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        return batch["actions"][..., 0, :]

    def _support_actions(self, batch, batch_actions):
        support_actions = batch.get("support_actions")
        if support_actions is None:
            support_actions = batch_actions[:, None, :]
        elif support_actions.ndim == 4:
            support_actions = jnp.reshape(
                support_actions,
                (support_actions.shape[0], support_actions.shape[1], -1),
            )
        support_weights = batch.get(
            "support_weights",
            jnp.ones((batch_actions.shape[0], support_actions.shape[1]), dtype=batch_actions.dtype),
        )
        return support_actions, support_weights

    def _weight_support_from_scores(self, support_weights, support_q, prefix="positive"):
        mode = self.config.get(f"{prefix}_weighting", "uniform")
        support_weights = jnp.clip(support_weights, min=0.0)
        valid = support_weights > 0.0
        support_count = support_q.shape[1]
        clip_min = float(self.config.get(f"{prefix}_weight_clip_min", 1.0e-6))
        clip_max = float(self.config.get(f"{prefix}_weight_clip_max", 1.0))
        temp = max(float(self.config.get(f"{prefix}_temp", 1.0)), 1.0e-6)

        if mode == "uniform":
            raw_weights = support_weights
        elif mode == "q_top1":
            masked_scores = jnp.where(valid, support_q, -jnp.inf)
            selected_idx = jnp.argmax(masked_scores, axis=1)
            raw_weights = jax.nn.one_hot(selected_idx, support_count, dtype=support_q.dtype)
        elif mode == "q_topk":
            topk = int(self.config.get(f"{prefix}_topk", 0))
            topk = support_count if topk <= 0 else min(max(1, topk), support_count)
            masked_scores = jnp.where(valid, support_q, -jnp.inf)
            _, top_indices = jax.lax.top_k(masked_scores, topk)
            top_mask = jnp.sum(
                jax.nn.one_hot(top_indices, support_count, dtype=support_q.dtype),
                axis=1,
            ) > 0.0
            if bool(self.config.get(f"{prefix}_include_first_support", False)):
                first_mask = jnp.zeros_like(top_mask).at[:, 0].set(True)
                top_mask = jnp.logical_or(top_mask, jnp.logical_and(first_mask, valid))
            logits = support_q / temp + jnp.log(jnp.clip(support_weights, min=clip_min))
            logits = jnp.where(jnp.logical_and(valid, top_mask), logits, -jnp.inf)
            raw_weights = jax.nn.softmax(logits, axis=1)
        elif mode in ("q_softmax", "oracle_softmax"):
            logits = support_q / temp + jnp.log(jnp.clip(support_weights, min=clip_min))
            logits = jnp.where(valid, logits, -jnp.inf)
            raw_weights = jax.nn.softmax(logits, axis=1)
        else:
            raise ValueError(f"Unsupported {prefix}_weighting: {mode}")

        raw_weights = jnp.where(
            raw_weights > 0.0,
            jnp.clip(raw_weights, min=clip_min, max=clip_max),
            0.0,
        )
        weights = _normalize_weights(raw_weights)
        if bool(self.config.get(f"{prefix}_stop_grad_weights", True)):
            weights = jax.lax.stop_gradient(weights)

        prob = _normalize_weights(weights)
        entropy = -jnp.sum(prob * jnp.log(jnp.clip(prob, min=1.0e-8)), axis=1)
        selected_idx = jnp.argmax(prob, axis=1)
        selected_q = jnp.take_along_axis(support_q, selected_idx[:, None], axis=1)[:, 0]
        info = {
            f"{prefix}_weight_entropy_mean": jnp.mean(entropy),
            f"{prefix}_effective_num_supports": jnp.mean(jnp.exp(entropy)),
            f"{prefix}_weight_max_mean": jnp.mean(jnp.max(prob, axis=1)),
            f"{prefix}_selected_q_mean": jnp.mean(selected_q),
            f"{prefix}_selected_q_std": jnp.std(selected_q),
        }
        return weights, info

    def _sample_particles_with_model(self, observations, rng, num_particles, model="actor"):
        noise_dim = self.config["noise_dim"]
        noises = jax.random.normal(
            rng,
            (*observations.shape[:-1], num_particles, noise_dim),
        )
        obs_rep = jnp.repeat(observations[..., None, :], num_particles, axis=-2)
        actions = self.network.select(model)(obs_rep, noises)
        return jnp.clip(actions, -1.0, 1.0)

    def _transport_step_size(self):
        step_size = jnp.asarray(float(self.config["transport_step_size"]), dtype=jnp.float32)
        after_size = float(self.config.get("transport_step_size_after", 0.0))
        schedule_step = int(self.config.get("transport_step_size_schedule_step", 0))
        if after_size <= 0.0 or schedule_step <= 0:
            return step_size
        after_size = jnp.asarray(after_size, dtype=jnp.float32)
        return jnp.where(self.network.step >= schedule_step, after_size, step_size)

    def _score_action_set(self, observations, actions, q_agg, pessimism_coef=0.0, critic="critic"):
        batch_size, num_actions = actions.shape[:2]
        obs_rep = jnp.repeat(observations[:, None, :], num_actions, axis=1)
        flat_obs = obs_rep.reshape(batch_size * num_actions, observations.shape[-1])
        flat_actions = actions.reshape(batch_size * num_actions, actions.shape[-1])
        q_values = self.network.select(critic)(flat_obs, flat_actions)
        if q_values.ndim == 1:
            return q_values.reshape(batch_size, num_actions)
        q_values = q_values.reshape(q_values.shape[0], batch_size, num_actions)
        return aggregate_q_values(q_values, q_agg, pessimism_coef=pessimism_coef)

    def critic_loss(self, batch, grad_params, rng):
        batch_actions = self._batch_actions(batch)
        batch_size = batch_actions.shape[0]
        rng, sample_rng = jax.random.split(rng)

        with_target_actor = bool(self.config["target_actor"])
        next_actor = "target_actor" if with_target_actor else "actor"
        next_actions = self._sample_particles_with_model(
            batch["next_observations"][..., -1, :],
            sample_rng,
            int(self.config["target_actor_num_samples"]),
            model=next_actor,
        )
        next_q_samples = self._score_action_set(
            batch["next_observations"][..., -1, :],
            next_actions,
            self.config["target_q_agg"],
            pessimism_coef=self.config["pessimism_coef"],
            critic="target_critic",
        )
        next_q = aggregate_sample_values(next_q_samples, self.config["target_action_agg"])
        target_q = batch["rewards"][..., -1] + (
            self.config["discount"] ** self.config["horizon_length"]
        ) * batch["masks"][..., -1] * next_q

        q = self.network.select("critic")(
            batch["observations"],
            batch_actions,
            params=grad_params,
        )
        valid = batch["valid"][..., -1]
        critic_loss = (jnp.square(q - target_q.reshape((1, batch_size))) * valid).mean()
        q_agg = aggregate_q_values(q, self.config["q_agg"], pessimism_coef=self.config["pessimism_coef"])

        conservative_loss = jnp.asarray(0.0, dtype=batch_actions.dtype)
        support_q_mean = jnp.asarray(0.0, dtype=batch_actions.dtype)
        support_ref_q_mean = jnp.asarray(0.0, dtype=batch_actions.dtype)
        if self.config["conservative_coef"] > 0.0 and self.config["n_conservative_actor_samples"] > 0:
            support_actions, support_weights = self._support_actions(batch, batch_actions)
            rng, proposal_rng = jax.random.split(rng)
            proposal_actions = jax.lax.stop_gradient(
                self._sample_particles_with_model(
                    batch["observations"],
                    proposal_rng,
                    int(self.config["n_conservative_actor_samples"]),
                )
            )
            proposal_q = self._score_action_set(
                batch["observations"],
                proposal_actions,
                self.config["conservative_q_agg"],
                pessimism_coef=self.config["pessimism_coef"],
            )
            support_q = self._score_action_set(
                batch["observations"],
                support_actions,
                self.config["support_q_agg"],
                pessimism_coef=self.config["pessimism_coef"],
            )
            support_ref_q = jnp.sum(
                support_q * _normalize_weights(support_weights),
                axis=1,
            )
            proposal_minus_data_q = proposal_q - support_ref_q[:, None]
            conservative_loss = jnp.maximum(
                proposal_minus_data_q + self.config["conservative_margin"],
                0.0,
            ).mean()
            support_q_mean = support_q.mean()
            support_ref_q_mean = support_ref_q.mean()

        loss = critic_loss + self.config["conservative_coef"] * conservative_loss
        return loss, {
            "critic_loss": critic_loss,
            "conservative_loss": conservative_loss,
            "q_mean": q_agg.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "q_ensemble_std_mean": jnp.std(q, axis=0).mean() if q.ndim > 1 else jnp.asarray(0.0),
            "target_next_q_mean": next_q.mean(),
            "target_next_q_std_mean": next_q_samples.std(axis=1).mean(),
            "target_q_mean": target_q.mean(),
            "target_q_max": target_q.max(),
            "target_q_min": target_q.min(),
            "support_q_mean": support_q_mean,
            "support_ref_q_mean": support_ref_q_mean,
        }

    def actor_loss(self, batch, grad_params, rng):
        batch_actions = self._batch_actions(batch)
        batch_size, action_dim = batch_actions.shape
        support_actions, support_weights = self._support_actions(batch, batch_actions)

        rng, gen_rng, actor_candidate_rng = jax.random.split(rng, 3)
        gen_per_obs = int(self.config["gen_per_obs"])
        noises = jax.random.normal(gen_rng, (batch_size, gen_per_obs, self.config["noise_dim"]))
        obs_rep = jnp.repeat(batch["observations"][:, None, :], gen_per_obs, axis=1)
        gen_actions = self.network.select("actor")(obs_rep, noises, params=grad_params)
        gen_actions = jnp.clip(gen_actions, -1.0, 1.0)

        flat_obs = obs_rep.reshape(batch_size * gen_per_obs, batch["observations"].shape[-1])
        flat_actions = jax.lax.stop_gradient(gen_actions).reshape(batch_size * gen_per_obs, action_dim)

        actor_update_mode = self.config.get("actor_update_mode", "decomposed")
        num_actor_candidates = int(self.config.get("positive_num_actor_candidates", 0))
        if actor_update_mode in ("positive_drift", "positive_drift_plus_residual") and num_actor_candidates > 0:
            actor_candidate_noises = jax.random.normal(
                actor_candidate_rng,
                (batch_size, num_actor_candidates, self.config["noise_dim"]),
            )
            actor_candidate_obs = jnp.repeat(batch["observations"][:, None, :], num_actor_candidates, axis=1)
            actor_candidate_actions = self.network.select("actor")(
                actor_candidate_obs,
                actor_candidate_noises,
                params=grad_params,
            )
            actor_candidate_actions = jax.lax.stop_gradient(jnp.clip(actor_candidate_actions, -1.0, 1.0))
            support_actions = jnp.concatenate([support_actions, actor_candidate_actions], axis=1)
            actor_candidate_weights = jnp.ones(
                (batch_size, num_actor_candidates),
                dtype=support_weights.dtype,
            )
            support_weights = jnp.concatenate([support_weights, actor_candidate_weights], axis=1)

        def q_sum(actions):
            q_values = self.network.select("target_critic")(flat_obs, actions)
            q = aggregate_q_values(
                q_values,
                self.config["actor_q_agg"],
                pessimism_coef=self.config["actor_pessimism_coef"],
            )
            return q.sum()

        if actor_update_mode in ("positive_drift", "positive_drift_plus_residual"):
            q_grad = jnp.zeros((batch_size, gen_per_obs, action_dim), dtype=gen_actions.dtype)
        else:
            q_grad = jax.grad(q_sum)(flat_actions).reshape(batch_size, gen_per_obs, action_dim)
            q_grad = q_grad * float(self.config.get("q_grad_scale", 1.0))
        gen_q_values = self.network.select("target_critic")(flat_obs, flat_actions)
        gen_q = aggregate_q_values(
            gen_q_values,
            self.config["actor_q_agg"],
            pessimism_coef=self.config["actor_pessimism_coef"],
        ).reshape(batch_size, gen_per_obs)

        support_weights = jax.lax.stop_gradient(support_weights)
        positive_weights = support_weights
        positive_weight_info = {}
        if actor_update_mode in ("positive_drift", "positive_drift_plus_residual"):
            support_q = self._score_action_set(
                batch["observations"],
                support_actions,
                self.config["positive_q_agg"],
                pessimism_coef=self.config["positive_pessimism_coef"],
                critic="target_critic",
            )
            positive_weights, positive_weight_info = self._weight_support_from_scores(
                support_weights,
                support_q,
                prefix="positive",
            )
        else:
            support_q = self._score_action_set(
                batch["observations"],
                support_actions,
                self.config["actor_q_agg"],
                pessimism_coef=self.config["actor_pessimism_coef"],
                critic="target_critic",
            )
        data_q = jnp.sum(support_q * _normalize_weights(positive_weights), axis=1)

        loss, info = offline_transport_drift_loss(
            gen=gen_actions,
            offline_actions=support_actions,
            offline_weights=positive_weights,
            positive_actions=support_actions,
            positive_weights=positive_weights,
            adv_grad=q_grad,
            tau=self.config["drift_tau"],
            beta=self.config["drift_beta"],
            lambda_pi=self.config["drift_lambda_pi"],
            kernel_bandwidth=self.config["kernel_bandwidth"],
            transport_step_size=self._transport_step_size(),
            normalize_q_grad=self.config["normalize_q_grad"],
            q_grad_clip_norm=self.config["q_grad_clip_norm"],
            exclude_self_kde=self.config["exclude_self_kde"],
            adaptive_bandwidth=self.config["adaptive_bandwidth"],
            bandwidth_quantile=self.config["bandwidth_quantile"],
            bandwidth_scale=self.config["bandwidth_scale"],
            min_bandwidth=self.config["min_bandwidth"],
            max_bandwidth=self.config["max_bandwidth"],
            kde_kernel=self.config["kde_kernel"],
            compact_kernel_eps=self.config["compact_kernel_eps"],
            compact_kernel_score_clip=self.config["compact_kernel_score_clip"],
            compact_kernel_min_neighbors=self.config["compact_kernel_min_neighbors"],
            compact_kernel_fallback=self.config["compact_kernel_fallback"],
            compact_fallback_bandwidth=self.config["compact_fallback_bandwidth"],
            actor_update_mode=actor_update_mode,
            positive_drift_type=self.config["positive_drift_type"],
        )
        actor_loss = loss.mean()
        support_mse = jnp.min(
            jnp.mean(jnp.square(gen_actions[:, 0, None, :] - support_actions), axis=-1),
            axis=-1,
        ).mean()

        info = {
            **info,
            "actor_loss": actor_loss,
            "gen_q_mean": gen_q.mean(),
            "data_q_mean": data_q.mean(),
            "data_advantage_mean": (data_q - gen_q.mean(axis=1)).mean(),
            "support_q_std_mean": support_q.std(axis=1).mean(),
            "support_mse": support_mse,
            "transport_step_size": self._transport_step_size(),
            **positive_weight_info,
        }
        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for key, value in critic_info.items():
            info[f"critic/{key}"] = value

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for key, value in actor_info.items():
            info[f"actor/{key}"] = value

        return critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1.0 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "critic")
        if agent.config["target_actor"]:
            agent.target_update(new_network, "actor")
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(self, observations, rng):
        single_observation = observations.ndim == len(self.config["ob_dims"])
        observations = observations[None] if single_observation else observations
        num_samples = int(self.config["best_of_n"])
        eval_actor = self.config.get("eval_actor", "actor")
        if eval_actor == "target_actor" and not bool(self.config.get("target_actor", False)):
            eval_actor = "actor"
        actions = self._sample_particles_with_model(observations, rng, num_samples, model=eval_actor)
        q = self._score_action_set(
            observations,
            actions,
            self.config["actor_q_agg"],
            pessimism_coef=self.config["actor_pessimism_coef"],
            critic=self.config.get("eval_critic", "critic"),
        )
        indices = jnp.argmax(q, axis=-1)
        batch_shape = indices.shape
        flat_indices = indices.reshape(-1)
        flat_actions = actions.reshape((-1, num_samples, actions.shape[-1]))
        selected = flat_actions[jnp.arange(flat_indices.shape[0]), flat_indices]
        selected = selected.reshape(batch_shape + (actions.shape[-1],))
        return selected[0] if single_observation else selected

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        if config["noise_dim"] is None:
            config["noise_dim"] = full_action_dim
        ex_noises = jnp.zeros(ex_observations.shape[:-1] + (config["noise_dim"],), dtype=ex_actions.dtype)

        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["value_layer_norm"],
            num_ensembles=config["num_qs"],
        )
        actor_def = TanhImplicitActor(
            action_dim=full_action_dim,
            hidden_dims=tuple(config["actor_hidden_dims"]),
            layer_norm=config["actor_layer_norm"],
        )
        network_info = dict(
            actor=(actor_def, (ex_observations, ex_noises)),
            target_actor=(copy.deepcopy(actor_def), (ex_observations, ex_noises)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )

        networks = {key: value[0] for key, value in network_info.items()}
        network_args = {key: value[1] for key, value in network_info.items()}
        network_def = ModuleDict(networks)
        if config["clip_grad"]:
            network_tx = optax.chain(
                optax.clip_by_global_norm(max_norm=config["clip_grad_norm"]),
                optax.adam(learning_rate=config["lr"]),
            )
        else:
            network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]
        params["modules_target_actor"] = params["modules_actor"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="drift",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            value_hidden_dims=(512, 512, 512, 512),
            value_layer_norm=True,
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=False,
            discount=0.99,
            tau=0.005,
            num_qs=10,
            q_agg="mean",
            target_q_agg="pessimistic",
            actor_q_agg="pessimistic",
            target_action_agg="mean",
            pessimism_coef=0.5,
            actor_pessimism_coef=0.5,
            target_actor=True,
            eval_actor="actor",
            eval_critic="critic",
            target_actor_num_samples=8,
            best_of_n=8,
            noise_dim=None,
            gen_per_obs=8,
            drift_tau=0.75,
            drift_beta=1.0,
            drift_lambda_pi=1.0,
            actor_update_mode="decomposed",
            positive_drift_type="score",
            positive_weighting="q_softmax",
            positive_q_source="learned",
            positive_q_agg="pessimistic",
            positive_pessimism_coef=0.5,
            positive_temp=0.5,
            positive_topk=0,
            positive_include_first_support=False,
            positive_num_actor_candidates=0,
            positive_stop_grad_weights=True,
            positive_weight_clip_min=1.0e-6,
            positive_weight_clip_max=1.0,
            q_grad_scale=1.0,
            kernel_bandwidth=0.25,
            transport_step_size=0.05,
            transport_step_size_after=0.0,
            transport_step_size_schedule_step=0,
            normalize_q_grad=False,
            q_grad_clip_norm=10.0,
            exclude_self_kde=True,
            adaptive_bandwidth=False,
            bandwidth_quantile=0.5,
            bandwidth_scale=1.0,
            min_bandwidth=1.0e-3,
            max_bandwidth=None,
            kde_kernel="gaussian",
            compact_kernel_eps=1.0e-6,
            compact_kernel_score_clip=100.0,
            compact_kernel_min_neighbors=1,
            compact_kernel_fallback="nearest",
            compact_fallback_bandwidth=0.0,
            conservative_coef=0.0,
            conservative_margin=0.0,
            n_conservative_actor_samples=0,
            conservative_q_agg="pessimistic",
            support_q_agg="pessimistic",
            clip_grad=True,
            clip_grad_norm=1.0,
        )
    )
    return config
