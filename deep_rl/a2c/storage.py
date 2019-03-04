from collections import namedtuple
from .core import RolloutBatch
import numpy as np

class RolloutStorage:
    def __init__(self, initial_observations, initial_states = []):
        self.num_processes = initial_observations.shape[0]

        self._terminals = self._last_terminals = np.zeros(shape = (self.num_processes,), dtype = np.bool)
        self._states = self._last_states = initial_states
        self._observations = self._last_observations = initial_observations

        self._batch = []

    def _transform_observation(self, observation):
        if observation.dtype == np.uint8:
            return observation.astype(np.float32) / 255.0
        else:
            return observation.astype(np.float32)

    @property
    def observations(self):
        return self._transform_observation(self._observations)

    @property
    def terminals(self):
        return self._terminals.astype(np.float32)

    @property
    def masks(self):
        return 1 - self.terminals

    @property
    def states(self):
        return self._states

    def insert(self, observations, actions, rewards, terminals, values, states):
        self._batch.append((self._observations, actions, values, rewards, terminals))
        self._observations = observations
        self._terminals = terminals
        self._states = states

    def batch(self, last_values, gamma):
        # Batch in time dimension
        b_observations, b_actions, b_values, b_rewards, b_terminals = list(map(lambda *x: np.stack(x, axis = 1), *self._batch))

        # Compute cummulative returns
        last_returns = (1.0 - b_terminals[:, -1]) * last_values
        b_returns = np.concatenate([np.zeros_like(b_rewards), np.expand_dims(last_returns, 1)], axis = 1)
        for n in reversed(range(len(self._batch))):
            b_returns[:, n] = b_rewards[:, n] + \
                gamma * (1.0 - b_terminals[:, n]) * b_returns[:, n + 1]

        # Compute RNN reset masks
        b_masks = (1 - np.concatenate([np.expand_dims(self._last_terminals, 1), b_terminals[:,:-1]], axis = 1))
        result = RolloutBatch(
            self._transform_observation(b_observations),
            b_returns[:, :-1].astype(np.float32), 
            b_actions, 
            b_masks.astype(np.float32),
            self._last_states
        )

        self._last_observations = self._observations
        self._last_states = self._states
        self._last_terminals = self._terminals
        self._batch = []
        return result