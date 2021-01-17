from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from gym.vector import AsyncVectorEnv
from functools import partial
from dataclasses import dataclass

from deep_rl.utils.environment import with_collect_reward_info, TorchWrapper
from deep_rl.utils import to_device, detach_all, Trainer
from deep_rl.utils.tensor import expand_time_dimension, stack_observations


@dataclass
class RolloutBatch:
    observations: Any
    actions: torch.Tensor
    returns: torch.Tensor
    masks: torch.Tensor
    states: Any

    def to(self, device):
        return to_device(self, device)


class RolloutStorage:
    def __init__(self, num_processes, initial_observations, initial_states=[]):
        self.num_processes = num_processes
        self._terminals = self._last_terminals = torch.zeros((self.num_processes,), dtype=torch.bool)
        self._states = self._last_states = initial_states
        self._observations = self._last_observations = initial_observations
        self._batch = []

    @property
    def observations(self):
        return self._observations

    @property
    def terminals(self):
        return self._terminals.float()

    @property
    def masks(self):
        return 1 - self.terminals

    @property
    def states(self):
        return self._states

    def insert(self, observations, actions, rewards, terminals, values, action_log_prob, states):
        self._batch.append(to_device((self._observations, actions, values, rewards, terminals), torch.device('cpu')))
        self._observations = observations
        self._terminals = terminals
        self._states = states

    def batch(self, last_values, gamma):
        last_values = last_values.cpu()
        # Batch in time dimension
        b_actions, b_values, b_rewards, b_terminals = [torch.stack([b[i] for b in self._batch], axis=1) for i in range(1, 5)]
        b_observations = stack_observations([o[0] for o in self._batch])

        # Compute cummulative returns
        b_terminals = b_terminals.float()
        last_returns = (1.0 - b_terminals[:, -1]) * last_values
        b_returns = torch.cat([torch.zeros_like(b_rewards), last_returns.view(last_returns.shape[0], 1, *last_returns.shape[1:])], 1)
        for n in reversed(range(len(self._batch))):
            b_returns[:, n] = b_rewards[:, n] + \
                gamma * (1.0 - b_terminals[:, n]) * b_returns[:, n + 1]

        # Compute RNN reset masks
        b_masks = 1 - torch.cat([self._last_terminals.view(self._last_terminals.shape[0], 1, *self._last_terminals.shape[1:]).float(), b_terminals[:, :-1]], axis=1)
        result = RolloutBatch(
            b_observations,
            b_actions,
            b_returns[:, :-1],
            b_masks,
            self._last_states
        )

        self._last_observations = self._observations
        self._last_states = self._states
        self._last_terminals = self._terminals
        self._batch = []
        return result


class PAAC(Trainer):
    DEFAULT_NAME = 'paac'
    RolloutStorage = RolloutStorage

    def __init__(self, model_fn, env_fn, *,
                 num_agents: int = 16,
                 num_steps: int = 128,
                 learning_rate: float = 7e-4,
                 gamma: float = 0.99,
                 value_coefficient: float = 1.0,
                 entropy_coefficient: float = 0.01,
                 max_grad_norm: float = 0.0, **kwargs):
        super().__init__(model_fn, learning_rate=learning_rate, **kwargs)
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.max_grad_norm = max_grad_norm
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.gamma = gamma
        self.env_fn = with_collect_reward_info(env_fn)

    def setup(self, stage):
        if stage == 'fit':
            assert self.num_agents % self.world_size == 0, 'Number of agents has to be divisible by the world size'
            global_rank = self.global_rank
            agents_per_process = self.num_agents // self.world_size
            self.env = TorchWrapper(AsyncVectorEnv([partial(self.env_fn, rank=i) for i in range(global_rank, global_rank + agents_per_process)]))
            if hasattr(self.model, 'initial_states'):
                self._get_initial_states = lambda x=agents_per_process: to_device(self.model.initial_states(x), self.current_device)
            else:
                self._get_initial_states = lambda x: None
            self.rollout_storage = self.RolloutStorage(agents_per_process, self.env.reset(), self._get_initial_states(agents_per_process))
        super().setup(stage)

    def compute_loss(self, batch: RolloutBatch) -> Tuple[torch.Tensor, Dict]:
        policy_logits, value, _ = self.model(batch.observations, batch.masks, batch.states)

        dist = torch.distributions.Categorical(logits=policy_logits)
        action_log_probs = dist.log_prob(batch.actions)
        dist_entropy = dist.entropy().mean()

        # Compute losses
        advantages = batch.returns - value.squeeze(-1)
        value_loss = 0.5 * advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()
        loss = value_loss * self.value_coefficient + \
            action_loss - \
            dist_entropy * self.entropy_coefficient
        return loss, dict(policy_loss=action_loss, value_loss=value_loss, entropy=dist_entropy)

    @torch.no_grad()
    def step_single(self, observations, masks, states):
        observations = expand_time_dimension(observations)
        masks = masks.view(-1, 1)

        policy_logits, value, states = self.model(observations, masks, states)
        dist = torch.distributions.Categorical(logits=policy_logits)
        action = dist.sample()
        action_log_probs = dist.log_prob(action)
        return action.squeeze(1).detach(), value.squeeze(1).squeeze(-1).detach(), action_log_probs.squeeze(1).detach(), detach_all(states)

    @torch.no_grad()
    def value_single(self, observations, masks, states):
        observations = expand_time_dimension(observations)
        masks = masks.view(-1, 1)

        _, value, states = self.model(observations, masks, states)
        return value.squeeze(1).squeeze(-1).detach(), detach_all(states)

    @torch.no_grad()
    def collect_experience(self):
        episode_lengths = []
        episode_returns = []
        for _ in range(self.num_steps):
            actions, values, action_log_prob, states = self.step_single(to_device(self.rollout_storage.observations, self.current_device), self.rollout_storage.masks.to(self.current_device), self.rollout_storage.states)
            actions = actions.cpu()
            values = values.cpu()
            action_log_prob = action_log_prob.cpu()

            # Take actions in env and look the results
            observations, rewards, terminals, infos = self.env.step(actions)

            # Collect true rewards
            for info in infos:
                if 'episode' in info.keys():
                    episode_lengths.append(info['episode']['l'])
                    episode_returns.append(info['episode']['r'])

            self.store_experience(observations, actions, rewards, terminals, values, action_log_prob, states)

        # Prepare next batch starting point
        return torch.tensor(episode_lengths, dtype=torch.int32), torch.tensor(episode_returns, dtype=torch.float32)

    def store_experience(self, *args):
        self.rollout_storage.insert(*args)

    @torch.no_grad()
    def sample_experience_batch(self) -> RolloutBatch:
        last_values, _ = self.value_single(to_device(self.rollout_storage.observations, self.current_device), self.rollout_storage.masks.to(self.current_device), self.rollout_storage.states)
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
        return self.num_steps
