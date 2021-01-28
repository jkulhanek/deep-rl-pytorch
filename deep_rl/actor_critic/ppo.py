import torch
import torch.nn as nn
import dataclasses
from dataclasses import dataclass
from functools import partial
from .a2c import PAAC, RolloutBatch as _RolloutBatch, RolloutStorage as _RolloutStorage
from .a2c import to_device, stack_observations


def split_batches(num_batches, batch_size, x):
    permutation = torch.randperm(batch_size)
    groups = torch.chunk(permutation, num_batches)

    def _split_chunks(x):
        if x is None:
            for g in groups:
                yield None
        elif isinstance(x, torch.Tensor):
            for g in groups:
                yield x[g, ...]
        elif isinstance(x, tuple):
            if len(x) == 0:
                for g in groups:
                    yield tuple()
            else:
                for y in zip(*map(_split_chunks, x)):
                    yield y
        elif isinstance(x, list):
            if len(x) == 0:
                for g in groups:
                    yield list()
            else:
                for y in zip(*map(_split_chunks, x)):
                    yield list(y)
        elif isinstance(x, dict):
            iterator_dict = {k: iter(_split_chunks(v)) for k, v in x.items()}
            for _ in groups:
                ret = dict()
                for k, iterator in iterator_dict.items():
                    ret[k] = next(iterator)
                yield ret
        elif dataclasses.is_dataclass(x):
            for y in _split_chunks(x.__dict__):
                yield dataclasses.replace(x, **y)
        else:
            raise ValueError(f'Type {type(x)} is not supported')
    return _split_chunks(x)


@dataclass
class RolloutBatch(_RolloutBatch):
    value: torch.Tensor
    action_log_probs: torch.Tensor
    advantages: torch.Tensor = None


class RolloutStorage(_RolloutStorage):
    def __init__(self, *args, gae_lambda=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.gae_lambda = gae_lambda

    def insert(self, observations, actions, rewards, terminals, values, action_log_probs, states):
        self._batch.append(to_device((self._observations, actions, values, rewards, action_log_probs, terminals), torch.device('cpu')))
        self._observations = observations
        self._terminals = terminals
        self._states = states

    def batch(self, last_values, gamma):
        gae_lambda = self.gae_lambda
        last_values = last_values.cpu()
        # Batch in time dimension
        b_actions, b_values, b_rewards, b_action_log_probs, b_terminals = [torch.stack([b[i] for b in self._batch], axis=1) for i in range(1, 6)]
        b_observations = stack_observations([o[0] for o in self._batch])

        # Compute cummulative returns
        b_terminals = b_terminals.float()
        b_values = torch.cat([b_values, last_values.unsqueeze(1)], 1)
        last_returns = (1.0 - b_terminals[:, -1]) * last_values
        b_returns = torch.cat([torch.zeros_like(b_rewards), last_returns.unsqueeze(1)], 1)
        gae = torch.zeros_like(last_values)
        for n in reversed(range(len(self._batch))):
            diff = b_rewards[:, n] + \
                gamma * (1.0 - b_terminals[:, n]) * b_values[:, n + 1] -\
                b_values[:, n]

            gae = diff + gamma * gae_lambda * (1.0 - b_terminals[:, n]) * gae
            b_returns[:, n] = gae + b_values[:, n]

        # Compute RNN reset masks
        b_masks = 1 - torch.cat([self._last_terminals.view(self._last_terminals.shape[0], 1, *self._last_terminals.shape[1:]).float(), b_terminals[:, :-1]], axis=1)
        result = RolloutBatch(
            b_observations,
            b_actions,
            b_returns[:, :-1],
            b_masks,
            self._last_states,
            b_values[:, :-1],
            b_action_log_probs,
        )

        self._last_observations = self._observations
        self._last_states = self._states
        self._last_terminals = self._terminals
        self._batch = []
        return result


class PPO(PAAC):
    DEFAULT_NAME = 'paac'
    RolloutStorage = RolloutStorage

    def __init__(self,
                 *args,
                 num_agents: int = 16,
                 num_steps: int = 128,
                 learning_rate: float = 2e-4,
                 value_coefficient: float = 0.5,
                 max_grad_norm: float = 0.5,
                 ppo_clip: float = 0.1,
                 ppo_epochs: int = 4,
                 gae_lambda: float = 0.95,
                 num_minibatches: int = 4,
                 **kwargs):
        super().__init__(*args,
                         num_agents=num_agents,
                         num_steps=num_steps,
                         learning_rate=learning_rate,
                         value_coefficient=value_coefficient,
                         max_grad_norm=max_grad_norm,
                         **kwargs)

        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.gae_lambda = gae_lambda
        self.RolloutStorage = partial(RolloutStorage, gae_lambda=gae_lambda)

    def configure_optimizers(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.learning_rate)

    def compute_loss(self, batch):
        policy_logits, value, _ = self.model(batch.observations, batch.masks, batch.states)
        value = value.view(value.shape[:-1])

        dist = torch.distributions.Categorical(logits=policy_logits)
        action_log_probs = dist.log_prob(batch.actions)
        dist_entropy = dist.entropy().mean()

        ratio = torch.exp(action_log_probs - batch.action_log_probs)
        surr1 = ratio * batch.advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip,
                            1.0 + self.ppo_clip) * batch.advantages
        action_loss = -torch.min(surr1, surr2).mean()

        value_pred_clipped = batch.value + \
            (value - batch.value).clamp(-self.ppo_clip, self.ppo_clip)
        value_loss = (value - batch.returns).pow(2)
        value_loss_clipped = (value_pred_clipped - batch.returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss,
                                     value_loss_clipped).mean()

        # Compute losses
        loss = value_loss * self.value_coefficient + \
            action_loss - \
            dist_entropy * self.entropy_coefficient
        return loss, dict(policy_loss=action_loss, value_loss=value_loss, entropy=dist_entropy)

    def training_step(self, batch):
        advantages = batch.returns - batch.value
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch.advantages = advantages
        for _ in range(self.ppo_epochs):
            generator = split_batches(self.num_minibatches, len(batch.returns), batch)
            for sub_batch in generator:
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

        # NOTE: the number of steps does not consider the number of grad updates
        return self.num_steps, None
