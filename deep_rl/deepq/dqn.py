import random
import time
import tqdm
import torch
from torch import nn

from deep_rl.utils import Trainer
from deep_rl.utils.environment import with_collect_reward_info
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
                 gamma: float = 0.9,
                 preload_steps: int = 8000,
                 adam_epsilon: float = 1.5*10e-4,
                 env_steps_per_batch: int = 4,
                 sync_frequency: int = 32000,
                 storage_size: int = 10**5,
                 prioritized_replay_alpha: float = 0.0,
                 prioritized_replay_beta: float = None,
                 epsilon: float = None,
                 use_doubling: bool = True,
                 seed: int = 42,
                 max_grad_norm: float = 0.0, **kwargs):
        super().__init__(model_fn, learning_rate=learning_rate, **kwargs)
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.adam_epsilon = adam_epsilon
        self.sync_frequency = sync_frequency
        self.env_fn = with_collect_reward_info(env_fn)
        self.env_steps_per_batch = env_steps_per_batch
        self.preload_steps = preload_steps
        self.storage_size = storage_size
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta = prioritized_replay_beta
        self.prioritized_replay_eps = 10e-6
        self.use_doubling = use_doubling
        self.rng_seed = seed
        self.rng = random.Random(self.rng_seed)
        if prioritized_replay_beta is None:
            self.prioritized_replay_beta = schedules.LinearSchedule(0.4, 1.0, None)
        if epsilon is None:
            epsilon = schedules.MultistepSchedule((0, schedules.LinearSchedule(1.0, 0.01, None)), (0.8, schedules.ConstSchedule(0.01)))
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
            self.env = self.env_fn()
            if self.prioritized_replay_alpha > 0:
                self.storage = PrioritizedReplayBuffer(self.storage_size, self.prioritized_replay_alpha, seed=self.rng_seed)
            else:
                self.storage = ReplayBuffer(self.storage_size, seed=self.rng_seed)
        super().setup(stage)

    def sample_random_action(self):
        return self.env.action_space.sample()

    def act(self, obs):
        return self.model(tensor_map(lambda x: x.unsqueeze(0)), obs).argmax().item()

    @torch.no_grad()
    def collect_experience(self):
        episode_lengths = []
        episode_returns = []
        num_steps = max(self.preload_steps - len(self.storage), self.env_steps_per_batch)
        step_iterator = range(num_steps)
        preloading = False
        if len(self.storage) < self.preload_steps:
            if self.global_rank == 0:
                step_iterator = tqdm.tqdm(step_iterator, desc='filling buffer')
            preloading = True
        for _ in step_iterator:
            if preloading or self.rng.rand() < self.epsilon:
                action = self.sample_random_action()
            else:
                action = self.model.act(self._state)
            state, reward, done, info = self.environment.step(action)
            self.store_experience(self._state, action, reward, done, state)
            self._state = state
            if done:
                self._state = self.environment.reset()

            episode_lengths = []
            episode_returns = []
            if 'episode' in info.keys():
                episode_lengths.append(info['episode']['l'])
                episode_returns.append(info['episode']['r'])
        if preloading:
            self._tstart = time.time()
        return torch.tensor(episode_lengths, dtype=torch.int32), torch.tensor(episode_returns, dtype=torch.float32)

    def store_experience(self, *args):
        self.storage.add(*args)

    @torch.no_grad()
    def sample_experience_batch(self) -> ReplayBatch:
        batch = self.storage.sample(self.minibatch_size, beta=self.prioritized_replay_beta)
        return batch

    def configure_optimizers(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.learning_rate)

    def compute_loss(self, batch):
        q = self.model(batch.observations)
        pcontinues = (1.0 - batch.dones.float()) * self.gamma
        if self.use_doubling:
            with torch.no_grad():
                q_next_selector = self.model_target(batch.next_observations)
                q_next_online_net = self.model(batch.next_observations)
            loss, tderr = double_qlearning(q, batch.actions, batch.rewards, pcontinues, q_next_online_net, q_next_selector, batch.weights)
        else:
            with torch.no_grad():
                q_next_online_net = self.model(batch.next_observations)
            loss, tderr = qlearning(q, batch.actions, batch.rewards, pcontinues, q_next_online_net, batch.weights)
        return loss.item(), tderr, dict()

    def training_step(self, batch):
        self._update_parameters()
        loss, tderr, metrics = self.compute_loss(*batch[:-1])
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
        return self.num_steps, tderr

    @torch.no_grad()
    def on_training_step_end(self, batch, output):
        tderr = output.detach().cpu()
        new_priorities = torch.abs(tderr) + self.prioritized_replay_eps
        self.storage.update_priorities(batch.idxes.detach().cpu(), new_priorities)
