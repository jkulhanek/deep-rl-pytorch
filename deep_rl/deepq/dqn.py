from typing import Optional
import random
import time
import tqdm
import torch
from torch import nn

from deep_rl.utils import Trainer
from deep_rl.utils.environment import with_collect_reward_info, TorchWrapper
from deep_rl.utils.tensor import tensor_map
from deep_rl import schedules
from .storage import ReplayBuffer, PrioritizedReplayBuffer, ReplayBatch


def qlearning(q, actions, rewards, pcontinues, q_next_online_net, weights=None):
    with torch.no_grad():
        target = rewards + pcontinues * torch.max(q_next_online_net, dim=-1)[0]

    actions = actions.long()
    q_actions = torch.gather(q, -1, actions.unsqueeze(-1)).squeeze(-1)

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target - q_actions
    loss = 0.5 * (td_error ** 2)
    if weights is not None:
        loss = loss * weights
    loss = loss.mean()
    return loss


def double_qlearning(q, actions, rewards, pcontinues, q_next, q_next_selector, weights=None):
    with torch.no_grad():
        indices = torch.argmax(q_next_selector, dim=-1, keepdim=True)
        target = rewards + pcontinues * torch.gather(q_next, -1, indices).squeeze(-1)

    actions = actions.long()
    q_actions = torch.gather(q, -1, actions.unsqueeze(-1)).squeeze(-1)

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target - q_actions
    loss = 0.5 * (td_error ** 2)
    if weights is not None:
        loss = loss * weights
    loss = loss.mean()
    return loss, td_error


def polyak_update(polyak_factor, target_network, network):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(polyak_factor*param.data + target_param.data*(1.0 - polyak_factor))


class DQN(Trainer):
    DEFAULT_NAME = 'dqn'

    def __init__(self, model_fn, env_fn, *,
                 learning_rate: float = 6.25e-5,
                 batch_size: int = 32,
                 gamma: float = 0.9,
                 preload_steps: int = 8000,
                 adam_epsilon: float = 1.5*10e-4,
                 env_steps_per_batch: int = 4,
                 sync_frequency: int = 32000,
                 storage_size: int = 10**5,
                 n_step_returns: int = 1,
                 prioritized_replay_alpha: float = 0.0,
                 prioritized_replay_beta: Optional[float] = None,
                 epsilon: float = None,
                 use_doubling: bool = True,
                 seed: int = 42,
                 max_grad_norm: float = 0.0, **kwargs):
        super().__init__(model_fn, learning_rate=learning_rate, **kwargs)
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.adam_epsilon = adam_epsilon
        self.sync_frequency = sync_frequency
        self.batch_size = batch_size
        self.env_fn = with_collect_reward_info(env_fn)
        self.env_steps_per_batch = env_steps_per_batch
        self.preload_steps = preload_steps
        self.storage_size = storage_size
        self.n_step_returns = n_step_returns
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta = prioritized_replay_beta
        self.prioritized_replay_eps = 10e-6
        self.use_doubling = use_doubling
        self.rng_seed = seed
        self.rng = random.Random(self.rng_seed)
        if prioritized_replay_beta is None:
            self.prioritized_replay_beta = schedules.LinearSchedule(0.4, 1.0, None)
        if epsilon is None:
            epsilon = schedules.MultistepSchedule((0, schedules.LinearSchedule(1.0, 0.01, None)), (0.8, schedules.ConstantSchedule(0.01)))
        self.epsilon = epsilon
        self._last_update = 0

    def _update_parameters(self):
        if not self.use_doubling:
            return
        if self.model_target is None:
            self.model_target = self.model_fn().to(self.current_device)
            self.model_target.load_state_dict(self.model.state_dict())
            self._last_update = self.global_t
        elif self.global_t - self._last_update >= self.sync_frequency:
            self.model_target.load_state_dict(self.model.state_dict())
            self._last_update = self.global_t

    def setup(self, stage):
        if stage == 'fit':
            self.env = TorchWrapper(self.env_fn())
            if self.prioritized_replay_alpha > 0:
                self.storage = PrioritizedReplayBuffer(self.storage_size, alpha=self.prioritized_replay_alpha, seed=self.rng_seed,
                                                       gamma=self.gamma, n_step_returns=self.n_step_returns)
            else:
                self.storage = ReplayBuffer(self.storage_size, seed=self.rng_seed, gamma=self.gamma, n_step_returns=self.n_step_returns)
            self._state = None
            self.model_target = None
        super().setup(stage)

    def sample_random_action(self):
        return torch.tensor(self.env.action_space.sample())

    def act(self, obs):
        return self.model(tensor_map(lambda x: x.unsqueeze(0), obs)).argmax()

    @torch.no_grad()
    def collect_experience(self):
        episode_lengths = []
        episode_returns = []
        num_steps = max(self.preload_steps - len(self.storage), self.env_steps_per_batch)
        step_iterator = range(num_steps)
        preloading = False
        if self._state is None:
            self._state = self.env.reset()
        if len(self.storage) < self.preload_steps:
            if self.global_rank == 0:
                step_iterator = tqdm.tqdm(step_iterator, desc='filling buffer')
            preloading = True
        for i in step_iterator:
            if preloading or self.rng.random() < self.epsilon:
                action = self.sample_random_action()
            else:
                action = self.act(self._state)
            state, reward, done, info = self.env.step(action)
            self.store_experience(self._state, action, reward, done)
            self._state = state
            if done:
                self._state = self.env.reset()

            episode_lengths = []
            episode_returns = []
            if 'episode' in info.keys():
                episode_lengths.append(info['episode']['l'])
                episode_returns.append(info['episode']['r'])
        if preloading:
            self._tstart = time.time()
        return i + 1, torch.tensor(episode_lengths, dtype=torch.int32), torch.tensor(episode_returns, dtype=torch.float32)

    def store_experience(self, *args):
        self.storage.add(*args)

    @torch.no_grad()
    def sample_experience_batch(self) -> ReplayBatch:
        if self.prioritized_replay_alpha > 0:
            batch = self.storage.sample(self.batch_size, beta=self.prioritized_replay_beta)
        else:
            batch = self.storage.sample(self.batch_size)
        return batch

    def configure_optimizers(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.learning_rate)

    def compute_loss(self, batch: ReplayBatch):
        # Computes the return estimates with bootstrapping
        with torch.no_grad():
            # self.model_target.reset_noise()
            q_next_online_net = self.model(batch.baseline_observations)
            indices = torch.argmax(q_next_online_net, dim=-1, keepdim=True)
            if self.use_doubling:
                q_next_selector = self.model_target(batch.baseline_observations)
            else:
                q_next_selector = q_next_online_net
            baselines = torch.gather(q_next_selector, -1, indices).squeeze(-1)
            return_estimates = batch.returns + batch.pcontinues * baselines

        # Selects actions
        q = self.model(batch.observations)
        q_actions = torch.gather(q, -1, batch.actions.unsqueeze(-1)).squeeze(-1)

        td_error = return_estimates - q_actions
        loss = 0.5 * (td_error ** 2)
        if batch.weights is not None:
            loss = loss * batch.weights
        loss = loss.mean()

        # Return new weights as td errors
        # We will reuse the tderr object not to waste memory
        new_weights = td_error.abs().add_(self.prioritized_replay_eps)
        return loss, new_weights, dict()

    def training_step(self, batch):
        self._update_parameters()
        loss, new_weights, metrics = self.compute_loss(batch)
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

        # Update
        return self.batch_size, new_weights

    @torch.no_grad()
    def on_training_step_end(self, batch, new_priorities):
        if self.prioritized_replay_alpha > 0:
            self.storage.update_priorities(batch.idxes.detach().cpu(), new_priorities.detach().cpu())

    def fit(self, *args, **kwargs):
        result = super().fit(*args, **kwargs)
        self.env.close()
        return result

    def evaluate(self, *args, **kwargs):
        result = super().evaluate(*args, **kwargs)
        self.env.close()
        return result
