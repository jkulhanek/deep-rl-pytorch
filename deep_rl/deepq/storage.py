from dataclasses import dataclass
import dataclasses
from typing import Any
import torch
from random import Random
import operator
from deep_rl.utils.tensor import stack_observations, from_numpy, to_device


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        return super(MinSegmentTree, self).reduce(start, end)


@dataclass
class ReplayBatch:
    observations: Any
    actions: torch.Tensor
    returns: torch.Tensor
    pcontinues: torch.Tensor
    baseline_observations: Any
    weights: torch.Tensor = None
    idxes: torch.Tensor = None

    def to(self, device):
        return to_device(self, device)


class ReplayBuffer:
    def __init__(self, size, gamma: float = 1.0, n_step_returns: int = 1, seed=42):
        self.rng = Random(seed)
        self.n_step_returns = n_step_returns
        self.gamma = gamma
        self._storage = []
        self._maxsize = size + n_step_returns
        self._next_idx = 0

    def __len__(self):
        return max(0, len(self._storage) - self.n_step_returns)

    @property
    def maxsize(self):
        return self._maxsize - self.n_step_returns + 1

    def _compute_returns(self, idxs):
        _data = [list(zip(*[self._storage[s + i] for i in range(self.n_step_returns)])) for s in idxs]
        _data = [(x[2:]) for x in _data]
        rewards, dones = tuple(zip(*_data))
        rewards = torch.tensor(rewards, dtype=torch.float32)
        terminals = torch.tensor(dones, dtype=torch.float32)
        returns = torch.zeros_like(rewards[:, 0].T)
        pcontinues = torch.ones_like(rewards[:, 0].T)
        for i in reversed(range(self.n_step_returns)):
            returns = rewards[:, i].T + self.gamma * (1 - terminals[:, i].T) * returns
            pcontinues = pcontinues * (1 - terminals[:, i].T)

        pcontinues *= self.gamma ** self.n_step_returns
        return returns, pcontinues

    def map_batch(self, idxes):
        batch = list(map(lambda x: self._storage[x], idxes))
        state, action, *_ = tuple(zip(*batch))
        state = stack_observations(state, axis=0)
        baseline_indexes = torch.tensor(idxes) + self.n_step_returns
        baseline_states = stack_observations([self._storage[x][0] for x in baseline_indexes], axis=0)
        action = torch.tensor(action, dtype=torch.int64)
        returns, pcontinues = self._compute_returns(idxes)
        return ReplayBatch(state, action, returns, pcontinues, baseline_states)

    def add(self, state, action, reward, done):
        data = (state, action, reward, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size) -> ReplayBatch:
        idxes = [self.rng.randrange(0, len(self._storage) - self.n_step_returns) for _ in range(batch_size)]
        return self.map_batch(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha=0.5, **kwargs):
        super().__init__(size, **kwargs)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - self.n_step_returns)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            valid = False
            while not valid:
                mass = self.rng.random() * every_range_len + i * every_range_len
                idx = self._it_sum.find_prefixsum_idx(mass)
                valid = (self._next_idx - idx) % self._maxsize >= self.n_step_returns
            res.append(idx)
        return res

    def sample(self, batch_size, beta) -> ReplayBatch:
        assert beta > 0
        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = torch.tensor(weights, dtype=torch.float32)
        encoded_sample = self.map_batch(idxes)
        idxes = torch.tensor(idxes, dtype=torch.int64)
        encoded_sample = dataclasses.replace(encoded_sample, weights=weights, idxes=idxes)
        return encoded_sample

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)
