from typing import Tuple, Dict
import numpy as np
import os
import time
import shutil

import torch
import torch.nn as nn
from gym.vector import AsyncVectorEnv
from functools import partial

from ..utils import detach_all, get_batch_size
from ..utils import batch_observations, expand_time_dimension

from deep_rl import metrics
from deep_rl.schedules import Schedule
from dataclasses import dataclass
from typing import Any, List, Tuple


def to_device(value, device):
    return value.to(device)


@dataclass
class RolloutBatch:
    observations: Any
    actions: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    values: torch.Tensor
    states: Any

    def to_device(self, device):
        return to_device(self, device)


class RolloutStorage:
    def __init__(self, initial_observations, initial_states=[]):
        self.num_processes = get_batch_size(initial_observations)

        self._terminals = self._last_terminals = np.zeros(shape=(self.num_processes,), dtype=np.bool)
        self._states = self._last_states = initial_states
        self._observations = self._last_observations = initial_observations

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
        b_actions, b_values, b_rewards, b_terminals = [torch.stack([b[i] for b in self._batch], axis=1) for i in range(1, 5)]
        b_observations = batch_observations([o[0] for o in self._batch])

        # Compute cummulative returns
        last_returns = (1.0 - b_terminals[:, -1]) * last_values
        b_returns = np.concatenate([torch.zeros_like(b_rewards), np.expand_dims(last_returns, 1)], axis=1)
        for n in reversed(range(len(self._batch))):
            b_returns[:, n] = b_rewards[:, n] + \
                gamma * (1.0 - b_terminals[:, n]) * b_returns[:, n + 1]

        # Compute RNN reset masks
        b_masks = (1 - np.concatenate([np.expand_dims(self._last_terminals, 1), b_terminals[:, :-1]], axis=1))
        result = (
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


def assert_has_collect_reward_info(env):
    current_env = env
    while hasattr(current_env, 'env'):
        if current_env.__class__.__name__ == 'RewardCollector':
            return
        current_env = env.env
    raise ValueError(f'Environment does not have RewardCollector wrapper')


def get_trainer_name(name, default='default'):
    if name is not None:
        return name
    if 'RL_NAME' in os.environ:
        return os.environ['RL_NAME']
    if 'WANDB_NAME' in os.environ:
        return os.environ['WANDB_NAME']
    if 'SLURM_JOB_NAME' in os.environ:
        return os.environ['SLURM_JOB_NAME']
    return default


def get_trainer_project(name, default='default'):
    if name is not None:
        return name
    if 'RL_PROJECT' in os.environ:
        return os.environ['RL_PROJECT']
    if 'WANDB_PROJECT' in os.environ:
        return os.environ['WANDB_PROJECT']
    return default


def find_save_path(path):
    if not os.path.exists(path):
        return path
    if not os.path.exists(os.path.join(path, 'v1')):
        os.makedirs(os.path.join(path, 'v1'))
        for fname in os.listdir(path):
            shutil.move(os.path.join(path, fname), os.path.join(path, 'v1', fname))

    i = 1
    while os.path.exists(os.path.join(path, f'v{i}')):
        i += 1
    return os.path.join(path, f'v{i}')


class Trainer:
    DEFAULT_NAME = 'default'

    def __init__(self, model_fn,
                 learning_rate: float,
                 loggers=None,
                 project: str = None,
                 name: str = None,
                 max_time_steps: int = None,
                 save_interval: int = 10000,
                 log_interval: int = 1000,
                 max_episodes: int = None):
        self.model_fn = model_fn
        self.model, self.optimizer = None, None
        self.global_rank = 0
        self.world_size = 1
        self.schedules = dict()
        self.name = get_trainer_name(name, self.DEFAULT_NAME)
        self.project = get_trainer_project(project)
        self.project = project
        self.max_time_steps = max_time_steps
        self.max_episodes = max_episodes
        self.learning_rate = learning_rate
        self.loggers = loggers or []

    def fit(self):
        self._setup('fit')
        while True:
            episode_lengths, returns = self.collect_experience()
            self.metrics['return'](returns)
            self.metrics['episode_length'](episode_lengths)

            # We multiply by the world size, since the reduction in distributed setting is mean
            self.metrics['episodes'](len(returns) * self.world_size)
            batch = self.sample_experience_batch()

            # Update learning rate with scheduler
            if 'learning_rate' in self.schedules:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            num_steps = self.training_step(batch)
            self.global_t += num_steps * self.world_size

            self.metrics['updates'](1)
            self.metrics['fps'](self.global_t / (time.time() - self._tstart))

            if self.global_t >= self.max_time_steps:
                break
            if self.global_episodes >= self.max_episodes:
                break
            if self.global_t - self._last_save_step >= self.save_interval:
                self.save()
                self._last_save_step = self.global_t
            if self.global_t - self._last_log_step >= self.log_interval:
                self._collect_logs()
                self._last_log_step = self.global_t

        self.save()

    def _collect_logs(self):
        logs = self.metrics.collect(self.world_size > 1)
        for logger in self.loggers:
            logger.log_metrics(self, logs, self.global_t)

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'weights.pth'))
        torch.save(dict(
            optimizer=self.optimizer.state_dict(),
            global_t=self.global_t
        ), os.path.join(self.save_path, 'checkpoint.pth'))

    def log(self, name, value):
        self.metrics[name](value)

    def _setup(self, stage):
        if self.model is None:
            self.model = self.model_fn()
        if stage == 'fit':
            self.global_t = 0
            if self.optimizer is None:
                self.optimizer = self.configure_optimizers(self.model)
            self._tstart = time.time()
            self._last_save
            self.metrics = metrics.MetricContext(
                updates=metrics.AccumulatedMetric(lambda x, y: x + y),
                fps=metrics.LastValue(),
                episodes=metrics.Mean(is_distributed=True))
            self._last_save_step = 0
            self._last_log_step = 0
            self.save_path = find_save_path(os.path.join(os.environ.get('MODELS_PATH', os.path.expanduser('~/models')), self.project, self.name))
            for logger in self.loggers:
                logger.setup(stage)
                if hasattr(logger.save_path):
                    self.save_path = logger.save_path
        self.setup(stage)

    def setup(stage):
        pass

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in self.schedules:
            self.schedules.pop(name)
        if isinstance(value, Schedule):
            self.schedules[name] = value

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if isinstance(value, Schedule):
            if not hasattr(self, 'global_t'):
                raise Exception('Schedules are supported only for classes with global_t property')
            value.step(getattr(self, 'global_t'))
            return value()
        else:
            return value

    def __delattr__(self, name):
        super().__delattr__(name)
        if name in self.schedules:
            self.schedules.pop(name)

    def collect_experience(self):
        raise NotImplementedError()

    def sample_experience_batch(self):
        raise NotImplementedError()

    def configure_optimizers(self, model):
        raise NotImplementedError()

    def training_step(self):
        raise NotImplementedError()


class A2C(Trainer):
    DEFAULT_NAME = 'paac'
    RolloutStorage = RolloutStorage

    def __init__(self, model_fn, env_fn,
            num_agents: int = 16,
            learning_rate: float = 7e-4,
            max_grad_norm: float = 0.0, **kwargs):
        super().__init__(model_fn, learning_rate=learning_rate, **kwargs)
        self.num_agents = num_agents
        self.max_grad_norm = max_grad_norm
        self.env_fn = env_fn

    def setup(self, stage):
        super().setup(stage)
        if stage == 'fit':
            assert self.num_agents % self.world_size == 0, 'Number of agents has to be divisible by the world size'
            assert_has_collect_reward_info(self.env_fn(rank=0))
            global_rank = self.global_rank
            agents_per_process = self.num_agents // self.world_size
            self.env = AsyncVectorEnv([partial(self.env_fn, rank=i) for i in range(global_rank, global_rank + agents_per_process)])
            self.rollout_storage = self.RolloutStorage(self.env.reset(), getattr(self.model, 'get_initial_states', lambda *args: None)(agents_per_process))

    def compute_loss(self, batch) -> Tuple[torch.Tensor, Dict]:
        policy_logits, value, _ = model(observations, masks, states)

        dist = torch.distributions.Categorical(logits=policy_logits)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()

        # Compute losses
        advantages = returns - value.squeeze(-1)
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()
        loss = value_loss * self.value_coefficient + \
            action_loss - \
            dist_entropy * self.entropy_coefficient
        return loss, dict(policy_loss=action_loss, value_loss=value_loss, entropy=dist_entropy)

    @torch.no_grad()
    def step_single(self, observations, masks, states):
        batch_size = get_batch_size(observations)
        observations = expand_time_dimension(observations)
        masks = masks.view(batch_size, 1)

        policy_logits, value, states = self.model(observations, masks, states)
        dist = torch.distributions.Categorical(logits=policy_logits)
        action = dist.sample()
        action_log_probs = dist.log_prob(action)
        return action.squeeze(1).detach(), value.squeeze(1).squeeze(-1).detach(), action_log_probs.squeeze(1).detach(), detach_all(states)

    @torch.no_grad()
    def value_single(self, observations, masks, states):
        batch_size = get_batch_size(observations)
        observations = expand_time_dimension(observations)
        masks = masks.view(batch_size, 1)

        _, value, states = self.model(observations, masks, states)
        return value.squeeze(1).squeeze(-1).detach(), detach_all(states)

    def collect_experience(self):
        episode_lengths = []
        episode_returns = []
        for _ in range(self.num_steps):
            actions, values, action_log_prob, states = self._step(self.rollout_storage.observations, self.rollout_storage.masks, self.rollout_storage.states)

            # Take actions in env and look the results
            observations, rewards, terminals, infos = self.env.step(actions)

            # Collect true rewards
            for info in infos:
                if 'episode' in info.keys():
                    episode_lengths.append(info['episode']['l'])
                    episode_returns.append(info['episode']['r'])

            self.rollout_storage.insert(observations, actions, rewards, terminals, values, states)

        # Prepare next batch starting point
        return torch.tensor(episode_lengths, dtype=torch.int32), torch.tensor(episode_returns, dtype=torch.float32)

    def sample_experience_batch(self):
        last_values, _ = self.value_single(self.rollout_storage.observations, self.rollout_storage.masks, self.rollout_storage.states)
        batched = self.rollout_storage.batch(last_values, self.gamma)

        # Prepare next batch starting point
        return batched

    def configure_optimizers(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.learning_rate)

    def training_step(self, batch):
        loss, metrics = self.compute_loss(batch)
        for k, v in metrics.items():
            self.log(k, v.item())
        self.log('loss', loss.item())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0.0:
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.log('grad_norm', grad_norm)
        self.optimizer.step()
