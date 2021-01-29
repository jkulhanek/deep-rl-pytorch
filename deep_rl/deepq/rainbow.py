import torch
from torch.nn import functional as F

from deep_rl import schedules
from .dqn import DQN


class Rainbow(DQN):
    DEFAULT_NAME = 'rainbow'

    def __init__(self, model_fn, env_fn, *,
                 learning_rate: float = 6.25e-5,
                 batch_size: int = 32,
                 preload_steps: int = 80000,
                 adam_epsilon: float = 1.5*10e-4,
                 env_steps_per_batch: int = 4,
                 sync_frequency: int = 32000,
                 storage_size: int = 10**5,
                 distributional: bool = True,
                 distributional_atoms: int = 51,
                 distributional_vmin: float = -10.0,
                 distributional_vmax: float = 10.0,
                 prioritized_replay_alpha: float = 0.5,
                 prioritized_replay_beta: float = None,
                 noisy_nets: bool = True,
                 use_doubling: bool = True,
                 epsilon: float = None, **kwargs):
        if prioritized_replay_beta is None:
            prioritized_replay_beta = schedules.LinearSchedule(0.4, 1.0, None)
        if epsilon is None:
            if noisy_nets:
                epsilon = 0.0
            else:
                epsilon = schedules.MultistepSchedule((0, schedules.LinearSchedule(1.0, 0.01, None)), (0.8, schedules.ConstantSchedule(0.01)))
        super().__init__(model_fn, env_fn,
                         learning_rate=learning_rate,
                         prioritized_replay_alpha=prioritized_replay_alpha,
                         prioritized_replay_beta=prioritized_replay_beta,
                         epsilon=epsilon,
                         storage_size=storage_size,
                         sync_frequency=sync_frequency,
                         env_steps_per_batch=env_steps_per_batch,
                         preload_steps=preload_steps,
                         batch_size=batch_size,
                         use_doubling=use_doubling, **kwargs)
        self.distributional = distributional
        self.distributional_atoms = distributional_atoms
        self.distributional_vmin = distributional_vmin
        self.distributional_vmax = distributional_vmax
        self.supports = None

    def setup(self, stage):
        super().setup(stage)
        if self.supports is None:
            self.supports = torch.arange(self.distributional_atoms, dtype=torch.float32) \
                .mul_((self.distributional_vmax - self.distributional_vmin) / (self.distributional_atoms - 1)) \
                .add_(self.distributional_vmin)
        self.supports = self.supports.to(self.current_device)

    def collect_experience(self, *args, **kwargs):
        if self.noisy_nets:
            self.model.reset_noise()
        return super().collect_experience(*args, **kwargs)

    def act(self, obs):
        if not self.distributional:
            super().act(obs)
        q = F.softmax(self.model(obs), -1) @ self.supports
        return q.argmax(-1)

    def compute_loss(self, batch):
        if self.noisy_nets:
            self.model_target.reset_noise()
        if not self.distributional:
            return super().compute_loss(batch)

        # Uses the C51 algorithm
        # Computes the distribution estimates with bootstrapping
        with torch.no_grad():
            target_probs = F.softmax(self.model_target(batch.next_observations), -1)
            q_next_online_net = F.softmax(self.model(batch.next_observations), -1) @ self.supports
            greedy_actions = torch.argmax(q_next_online_net, dim=-1, keepdim=False)
            d_probs = target_probs[range(target_probs.shape[0]), greedy_actions, :]
            d_values = batch.returns.view(-1, 1) + batch.pcontinues.view(-1, 1) * self.gamma ** self.n_step_returns * self.supports

            # Project d_probs to d_probs_projected with L2 projection
            delta_z = (self.distributional_vmax - self.distributional_vmin) / (self.distributional_atoms - 1)
            d_values.clamp_(self.distributional_vmin, self.distributional_vmax)
            d_values.add_(-self.distributional_vmin).mul_(1/delta_z)
            b = d_values
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix the problem when b is integer and the probability mass is dissapearing
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.distributional_atoms - 1)) * (l == u)] += 1
            # Build d_probs_projected
            # Transform to linear array to save allocated space
            d_probs_projected = torch.zeros_like(d_probs)
            offset = torch.linspace(0, (b.shape[0] - 1) * self.distributional_atoms, b.shape[0]).unsqueeze(1).long().to(b.device).view(b.shape[0], -1)
            d_probs_projected.view(-1).index_add_(0, (l + offset).view(-1), (d_probs * (u.float() - b)).view(-1))
            d_probs_projected.view(-1).index_add_(0, (u + offset).view(-1), (d_probs * (b - l.float())).view(-1))

        # Selects actions
        d = F.log_softmax(self.model(batch.observations), -1)
        d_actions = d[range(d.shape[0]), batch.actions, :]
        kldiv = -(d_probs_projected * d_actions).sum(-1)
        loss = kldiv

        if batch.weights is not None:
            loss = loss * batch.weights
        loss = loss.mean()
        return loss.mean(), kldiv.detach()
