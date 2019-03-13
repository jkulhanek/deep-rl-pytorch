from collections import namedtuple
import numpy as np

from ..common.pytorch import RolloutBatch

def batch_observations(observations):
    if isinstance(observations[0], tuple):
        return tuple(batch_observations(list(map(list, observations))))
    elif isinstance(observations[0], list):
        return list(map(lambda *x: batch_observations(x), *observations))
    elif isinstance(observations[0], dict):
        return {key: batch_observations([o[key] for o in observations]) for key in observations[0].keys()}
    else:
        return np.stack(observations, axis = 0)

class RolloutStorage:
    def __init__(self, initial_observation, initial_states = []):
        self._terminal = self._last_terminal = False
        self._states = self._last_states = initial_states
        self._observation = self._last_observation = initial_observation
        self._batch = []

    def _transform_observation(self, observation):
        if isinstance(observation, np.ndarray):
            if observation.dtype == np.uint8:
                return observation.astype(np.float32) / 255.0
            else:
                return observation.astype(np.float32)
        elif isinstance(observation, list):
            return [self._transform_observation(x) for x in observation]
        elif isinstance(observation, tuple):
            return tuple([self._transform_observation(x) for x in observation])
        

    @property
    def observation(self):
        return self._transform_observation(self._observation)

    @property
    def terminal(self):
        return float(self._terminal)

    @property
    def mask(self):
        return 1 - self._terminal

    @property
    def states(self):
        return self._states

    def insert(self, observation, action, reward, terminal, value, states):
        self._batch.append((self._observation, action, value, reward, terminal))
        self._observation = observation
        self._terminal = terminal
        self._states = states

    def batch(self, last_value, gamma):
        # Batch in time dimension
        b_actions, b_values, b_rewards, b_terminals = [np.stack([b[i] for b in self._batch], axis = 0) for i in range(1, 5)]
        b_observations = batch_observations([o[0] for o in self._batch])

        # Compute cummulative returns
        last_returns = (1.0 - b_terminals[-1]) * last_value
        b_returns = np.concatenate([np.zeros_like(b_rewards), np.array([last_returns], dtype = np.float32)], axis = 0)
        for n in reversed(range(len(self._batch))):
            b_returns[n] = b_rewards[n] + \
                gamma * (1.0 - b_terminals[n]) * b_returns[n + 1]

        # Compute RNN reset masks
        b_masks = (1 - np.concatenate([np.array([self._last_terminal], dtype = np.bool), b_terminals[:-1]], axis = 0))
        result = RolloutBatch(
            self._transform_observation(b_observations),
            b_returns[:-1].astype(np.float32), 
            b_actions, 
            b_masks.astype(np.float32),
            self._last_states
        )

        self._last_observation = self._observation
        self._last_states = self._states
        self._last_terminal = self._terminal
        self._batch = []
        return result