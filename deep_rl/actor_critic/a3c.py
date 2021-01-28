from functools import partial
import torch
import gym
from torch import nn
import torch.multiprocessing as mp

import time
import tqdm

from deep_rl.utils.optim.shared_rmsprop import SharedRMSprop
from deep_rl.utils.trainer import Trainer, ScheduledMixin
from deep_rl.utils.environment import with_collect_reward_info, VectorTorchWrapper
from deep_rl.utils import to_device
from deep_rl.utils.tensor import get_batch_size
from deep_rl import metrics

from .a2c import PAAC


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __call__(self, *args, **kwargs):
        return self.x(*args, **kwargs)

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def serialize_function(fun):
    return CloudpickleWrapper(fun)


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


class A3C(PAAC):
    DEFAULT_NAME = 'a3c'

    def __init__(self, model_fn, env_fn, *,
                 num_agents: int = 16,
                 num_steps: int = 128,
                 rms_alpha: float = 0.99,
                 rms_epsilon: float = 1e-5,
                 learning_rate: float = 7e-4,
                 gamma: float = 0.99,
                 value_coefficient: float = 1.0,
                 entropy_coefficient: float = 0.01,
                 max_grad_norm: float = 40.0, **kwargs):
        # The parent PAAC is used only to copy useful methods
        Trainer.__init__(self, model_fn, learning_rate=learning_rate, **kwargs)
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.rms_alpha = rms_alpha
        self.rms_epsilon = rms_epsilon
        self.max_grad_norm = max_grad_norm
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.gamma = gamma
        self.env_fn = with_collect_reward_info(env_fn)
        self._is_worker = False

    def setup(self, stage):
        if stage == 'fit' and not self._is_worker:
            assert self.world_size == 1, 'A3C does not support distributed training'
            self.model = self.model.share_memory()
        if stage == 'fit' and self._is_worker:
            self.model = self.model_fn().to(self.current_device)
            self._tstart = time.time()
            self.metrics = metrics.MetricsContext(**{
                'return': metrics.Mean(),
                'updates': metrics.AccumulatedMetric(lambda x, y: x + y),
                'fps': metrics.LastValue(),
                'episodes': metrics.Mean(is_distributed=False)})

        if stage == 'fit' and self._is_worker or stage == 'evaluate':
            self.env = VectorTorchWrapper(gym.vector.SyncVectorEnv([partial(self.env_fn, self.global_rank)]))
            if hasattr(self.model, 'initial_states'):
                self._get_initial_states = lambda x: to_device(self.model.initial_states(x), self.current_device)
            else:
                self._get_initial_states = lambda x: None
            self.rollout_storage = self.RolloutStorage(1, self.env.reset(), self._get_initial_states(1))

    def configure_optimizers(self, model):
        return SharedRMSprop(self.model.parameters(), self.learning_rate, self.rms_alpha, self.rms_epsilon).share_memory()

    @property
    def current_device(self):
        return torch.device('cpu')

    @torch.no_grad()
    def collect_experience(self):
        episode_lengths = []
        episode_returns = []
        for n in range(self.num_steps):
            actions, values, action_log_prob, states = self.step_single(to_device(self.rollout_storage.observations, self.current_device),
                                                                        self.rollout_storage.masks.to(self.current_device), self.rollout_storage.states)
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
            if terminals.sum().item() > 0:
                break

        # Prepare next batch starting point
        return n + 1, torch.tensor(episode_lengths, dtype=torch.int32), torch.tensor(episode_returns, dtype=torch.float32)

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
        ensure_shared_grads(self.model, self.shared_model)
        self.optimizer.step()

    def _worker_run(self):
        num_agents, self.num_agents = self.num_agents, 1
        self.setup('fit')
        self.num_agents = num_agents
        while True:
            # self.model.load_state_dict(self.shared_model.state_dict())
            _, episode_lengths, returns = self.collect_experience()
            batch = self.sample_experience_batch()

            t_diff = get_batch_size(batch, axis=1)
            batch = batch.to(self.current_device)

            # Update learning rate with scheduler
            if 'learning_rate' in self.schedules:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            self.training_step(batch)
            self.metrics['return'](returns)
            self.metrics['episode_length'](episode_lengths)
            self.metrics['episodes'](len(returns))
            self.metrics['updates'](1)
            self.metrics['fps'](self.global_t / (time.time() - self._tstart))
            self.shared_queue.put((t_diff, self.metrics.collect(full_stats=True)))

            # Exit gracefully
            if self.max_time_steps is not None and self.global_t >= self.max_time_steps:
                break
            if self.max_episodes is not None and self.global_episodes >= self.max_episodes:
                break
        self.env.close()

    @property
    def global_t(self):
        return self.shared_global_t.value

    @global_t.setter
    def global_t(self, value):
        assert self.global_rank == 0
        if hasattr(self, 'shared_global_t'):
            self.shared_global_t.value = value

    def _get_worker_state_dict(self):
        state_dict = self.state_dict()
        del state_dict['optimizer']
        del state_dict['model']
        del state_dict['loggers']
        state_dict['optimizer'] = self.optimizer
        state_dict['shared_model'] = self.model
        state_dict['shared_global_t'] = self.shared_global_t
        state_dict['shared_queue'] = self.shared_queue
        state_dict['model_fn'] = serialize_function(self.model_fn)
        state_dict['env_fn'] = serialize_function(self.env_fn)
        return state_dict

    def fit(self):
        model_fn = self.model_fn
        self.model_fn = lambda: model_fn().share_memory()
        self._setup('fit')
        self.model_fn = model_fn

        # Initialize globals
        smp = torch.multiprocessing.get_context('spawn')
        self.shared_queue = smp.SimpleQueue()
        self.shared_global_t = smp.Value('i', 0)
        state_dict = self._get_worker_state_dict()

        # Spawn processes
        ctx = torch.multiprocessing.spawn(_worker_run, args=(self.__class__, state_dict), nprocs=self.num_agents, join=False)
        progress = tqdm.tqdm(desc='training', total=self.max_time_steps)
        while not ctx.join(0.02):
            while not self.shared_queue.empty():
                message = self.shared_queue.get()
                delta_t, metrics = message
                for m, v in metrics.items():
                    if isinstance(v, tuple):
                        self.metrics[m](*v)
                    else:
                        self.metrics[m](v)
                self.shared_global_t.value += delta_t
                progress.update(delta_t)

                if self.global_t - self._last_save_step >= self.save_interval:
                    self.save()
                    self._last_save_step = self.global_t
                if self.global_t - self._last_log_step >= self.log_interval:
                    metrics = self._collect_logs()
                    self._last_log_step = self.global_t
                    self._last_log_step = self.global_t
                    postfix = 'return: {return:.2f}'.format(**metrics)
                    if 'loss' in metrics:
                        postfix += ', loss: {loss:.4f}'.format(**metrics)
                    progress.set_postfix_str(postfix)
        progress.close()
        self.save()

    @classmethod
    def build_worker(cls, state_dict):
        self = cls.__new__(cls)
        ScheduledMixin.__init__(self)
        self._is_worker = True
        self.load_state_dict(state_dict)
        return self


def _worker_run(global_rank, cls, state_dict):
    algo = cls.build_worker(state_dict)
    algo.global_rank = global_rank
    algo._worker_run()
