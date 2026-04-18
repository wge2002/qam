import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))

class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        self.behavior_support_enabled = False
        self.behavior_support_k = 1
        self.behavior_support_actual_k = 1
        self.behavior_support_temperature = 1.0
        self.behavior_support_include_self = True
        self.behavior_support_chunk_size = 65536
        self.behavior_support_action_chunking = False
        self.normalized_observations = None
        self.normalized_obs_sq = None

    def configure_behavior_support(
        self,
        k_support=1,
        temperature=1.0,
        include_self=True,
        normalize_observations=True,
        chunk_size=65536,
        action_chunking=False,
    ):
        """Enable raw-observation KNN behavior support for drift-style losses."""
        self.behavior_support_k = max(1, int(k_support))
        self.behavior_support_temperature = max(float(temperature), 1.0e-8)
        self.behavior_support_include_self = bool(include_self)
        self.behavior_support_chunk_size = max(1, int(chunk_size))
        self.behavior_support_action_chunking = bool(action_chunking)
        max_support = self.size if self.behavior_support_include_self else self.size - 1
        self.behavior_support_actual_k = min(self.behavior_support_k, max_support)
        if self.behavior_support_actual_k < 1:
            raise ValueError("Cannot build behavior support with include_self=False and size <= 1")

        observations = np.asarray(self['observations'], dtype=np.float32)
        if normalize_observations:
            obs_mean = observations.mean(axis=0, keepdims=True).astype(np.float32)
            obs_std = observations.std(axis=0, keepdims=True).astype(np.float32)
            obs_std = np.maximum(obs_std, 1.0e-6)
            observations = (observations - obs_mean) / obs_std
        self.normalized_observations = observations.astype(np.float32)
        self.normalized_obs_sq = np.sum(
            self.normalized_observations * self.normalized_observations,
            axis=1,
        ).astype(np.float32)
        self.behavior_support_enabled = True
        return self

    def get_random_idxs(self, num_idxs):
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        return batch

    def sample_sequence(self, batch_size, sequence_length, discount):
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
        
        data = {k: v[idxs] for k, v in self.items()}

        rewards = np.zeros(data['rewards'].shape + (sequence_length,), dtype=float)
        masks = np.ones(data['masks'].shape + (sequence_length,), dtype=float)
        valid = np.ones(data['masks'].shape + (sequence_length,), dtype=float)
        observations = np.zeros(data['observations'].shape[:-1] + (sequence_length, data['observations'].shape[-1]), dtype=float)
        next_observations = np.zeros(data['observations'].shape[:-1] + (sequence_length, data['observations'].shape[-1]), dtype=float)
        actions = np.zeros(data['actions'].shape[:-1] + (sequence_length, data['actions'].shape[-1]), dtype=float)
        terminals = np.zeros(data['terminals'].shape + (sequence_length,), dtype=float)

        for i in range(sequence_length):
            cur_idxs = idxs + i

            if i == 0:
                rewards[..., 0] = self['rewards'][cur_idxs]
                masks[..., 0] = self["masks"][cur_idxs]
                terminals[..., 0] = self["terminals"][cur_idxs]
            else:
                valid[..., i] = (1.0 - terminals[..., i - 1])
                rewards[..., i] = rewards[..., i - 1] + (self['rewards'][cur_idxs] * (discount ** i) * valid[..., i])
                masks[..., i] = np.minimum(masks[..., i-1], self["masks"][cur_idxs]) * valid[..., i] + masks[..., i-1] * (1. - valid[..., i])
                terminals[..., i] = np.maximum(terminals[..., i-1], self["terminals"][cur_idxs])
            
            actions[..., i, :] = self['actions'][cur_idxs]
            next_observations[..., i, :] = self['next_observations'][cur_idxs] * valid[..., i:i+1] + next_observations[..., i-1, :] * (1. - valid[..., i:i+1])
            observations[..., i, :] = self['observations'][cur_idxs]
            
        batch = dict(
            observations=data['observations'].copy(),
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
        )
        batch.update(self._behavior_support(idxs, sequence_length))
        return batch

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        return result

    def _support_weights_from_dist2(self, support_dist2):
        logits = -support_dist2 / self.behavior_support_temperature
        logits = logits - logits.max(axis=1, keepdims=True)
        support_weights = np.exp(logits).astype(np.float32)
        return support_weights / np.maximum(
            support_weights.sum(axis=1, keepdims=True),
            1.0e-8,
        )

    def _knn_search(self, idx):
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        queries = self.normalized_observations[idx]
        query_sq = np.sum(queries * queries, axis=1).astype(np.float32)
        k = self.behavior_support_actual_k
        best_dist2 = np.full((idx.shape[0], k), np.inf, dtype=np.float32)
        best_idx = np.full((idx.shape[0], k), -1, dtype=np.int64)
        rows = np.arange(idx.shape[0])

        for start in range(0, self.size, self.behavior_support_chunk_size):
            end = min(start + self.behavior_support_chunk_size, self.size)
            refs = self.normalized_observations[start:end]
            dist2 = query_sq[:, None] + self.normalized_obs_sq[start:end][None, :]
            dist2 = dist2 - 2.0 * np.matmul(queries, refs.T)
            np.maximum(dist2, 0.0, out=dist2)

            if not self.behavior_support_include_self:
                in_chunk = (idx >= start) & (idx < end)
                if np.any(in_chunk):
                    dist2[rows[in_chunk], idx[in_chunk] - start] = np.inf

            chunk_k = min(k, end - start)
            chunk_part = np.argpartition(dist2, kth=chunk_k - 1, axis=1)[:, :chunk_k]
            chunk_dist2 = np.take_along_axis(dist2, chunk_part, axis=1)
            chunk_idx = chunk_part.astype(np.int64) + start

            candidate_dist2 = np.concatenate((best_dist2, chunk_dist2), axis=1)
            candidate_idx = np.concatenate((best_idx, chunk_idx), axis=1)
            merge_part = np.argpartition(candidate_dist2, kth=k - 1, axis=1)[:, :k]
            best_dist2 = np.take_along_axis(candidate_dist2, merge_part, axis=1)
            best_idx = np.take_along_axis(candidate_idx, merge_part, axis=1)

        order = np.argsort(best_dist2, axis=1)
        support_dist2 = np.take_along_axis(best_dist2, order, axis=1)
        support_idx = np.take_along_axis(best_idx, order, axis=1)
        return support_idx.astype(np.int64), support_dist2.astype(np.float32)

    def _support_action_sequence(self, support_idx, sequence_length):
        support_idx = np.asarray(support_idx, dtype=np.int64)
        if not self.behavior_support_action_chunking:
            return self['actions'][support_idx]

        actions = np.zeros(
            support_idx.shape + (sequence_length, self['actions'].shape[-1]),
            dtype=self['actions'].dtype,
        )
        for i in range(sequence_length):
            cur_idxs = np.minimum(support_idx + i, self.size - 1)
            actions[..., i, :] = self['actions'][cur_idxs]
        return actions

    def _behavior_support(self, idxs, sequence_length):
        if not self.behavior_support_enabled:
            return {}

        idxs = np.asarray(idxs, dtype=np.int64).reshape(-1)
        if self.behavior_support_actual_k == 1 and self.behavior_support_include_self:
            support_idx = idxs[:, None]
            support_weights = np.ones((idxs.shape[0], 1), dtype=np.float32)
        else:
            support_idx, support_dist2 = self._knn_search(idxs)
            support_weights = self._support_weights_from_dist2(support_dist2)

        return {
            "support_idx": support_idx,
            "support_actions": self._support_action_sequence(support_idx, sequence_length),
            "support_weights": support_weights.astype(np.float32),
        }

class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, pointer=0, size=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = size
        self.pointer = pointer

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0
