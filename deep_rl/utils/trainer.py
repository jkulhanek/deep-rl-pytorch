import os
import shutil
import torch
import time
from deep_rl import metrics
from deep_rl.schedules import Schedule
from deep_rl.utils import logging
from .logging import rank_zero_only
import tqdm


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


def find_save_dir(path):
    if not os.path.exists(path):
        return path
    if not os.path.exists(os.path.join(path, 'v1')):
        os.makedirs(os.path.join(path, 'v1'))
        for fname in os.listdir(path):
            if fname == 'v1':
                continue
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
                 num_gpus: int = None,
                 max_episodes: int = None):
        object.__setattr__(self, 'schedules', dict())
        self.model_fn = model_fn
        self.model, self.optimizer = None, None
        self.global_rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.name = get_trainer_name(name, self.DEFAULT_NAME)
        self.project = get_trainer_project(project)
        self.project = project
        self.max_time_steps = max_time_steps
        self.max_episodes = max_episodes
        self.learning_rate = learning_rate
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.loggers = loggers if loggers is not None else [logging.TensorBoardLogger()]
        self.num_gpus = num_gpus if num_gpus is not None else torch.cuda.device_count()
        assert self.num_gpus < 2, "Distributed training is not yet supported"

    def fit(self):
        self._setup('fit')
        progress = tqdm.tqdm(desc='training', total=self.max_time_steps)
        while True:
            episode_lengths, returns = self.collect_experience()
            self.metrics['return'](returns)
            self.metrics['episode_length'](episode_lengths)

            # We multiply by the world size, since the reduction in distributed setting is mean
            self.metrics['episodes'](len(returns) * self.world_size)
            batch = self.sample_experience_batch()
            batch = batch.to(self.current_device)

            # Update learning rate with scheduler
            if 'learning_rate' in self.schedules:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            num_steps = self.training_step(batch)
            self.global_t += num_steps * self.world_size
            progress.update(num_steps * self.world_size)
            postfix = 'return: {return:.2f}'.format(**self.metrics)
            if 'loss' in self.metrics:
                postfix += ', loss: {loss:.4f}'.format(**self.metrics)
            progress.set_postfix_str(postfix)

            self.metrics['updates'](1)
            self.metrics['fps'](self.global_t / (time.time() - self._tstart))

            if self.max_time_steps is not None and self.global_t >= self.max_time_steps:
                break
            if self.max_episodes is not None and self.global_episodes >= self.max_episodes:
                break
            if self.global_t - self._last_save_step >= self.save_interval:
                self.save()
                self._last_save_step = self.global_t
            if self.global_t - self._last_log_step >= self.log_interval:
                self._collect_logs()
                self._last_log_step = self.global_t

        self.save()
        progress.close()

    @property
    def current_device(self):
        if self.num_gpus > 0:
            return torch.device('cuda', self.local_rank)
        else:
            return torch.device('cpu')

    def _collect_logs(self):
        logs = self.metrics.collect(self.world_size > 1)
        for logger in self.loggers:
            logger.log_metrics(logs, self.global_t)
        return logs

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'weights.pth'))
        torch.save(dict(
            optimizer=self.optimizer.state_dict(),
            global_t=self.global_t
        ), os.path.join(self.save_dir, 'checkpoint.pth'))
        for logger in self.loggers:
            logger.save()

    def log(self, name, value):
        self.metrics[name](value)

    def _setup(self, stage):
        if self.model is None:
            self.model = self.model_fn()

        # First, we will initialize the distributed training
        # if self.world_size > 1:

        #     self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank])
        # else:
        setattr(rank_zero_only, 'rank', self.global_rank)
        self.model.to(self.current_device)
        if stage == 'fit':
            self.global_t = 0
            if self.optimizer is None:
                self.optimizer = self.configure_optimizers(self.model)
            self._tstart = time.time()
            self.metrics = metrics.MetricsContext(
                updates=metrics.AccumulatedMetric(lambda x, y: x + y),
                fps=metrics.LastValue(),
                episodes=metrics.Mean(is_distributed=True))
            self._last_save_step = 0
            self._last_log_step = 0
            self.save_dir = None
            for logger in self.loggers:
                logger.setup(stage, self)
                if hasattr(logger, 'save_dir') and logger.save_dir is not None:
                    self.save_dir = logger.save_dir
            if self.save_dir is None:
                self.save_dir = find_save_dir(os.path.join(os.environ.get('MODELS_PATH', os.path.expanduser('~/models')), self.project, self.name))
            os.makedirs(self.save_dir, exist_ok=True)
            for logger in self.loggers:
                if not hasattr(logger, 'save_dir'):
                    continue
                property_obj = getattr(type(logger), 'save_dir', None)
                if isinstance(property_obj, property) and property_obj.fset is None:
                    continue
                logger.save_dir = self.save_dir

        # Synchronize initial weights
        self.setup(stage)

    def setup(self, stage):
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
