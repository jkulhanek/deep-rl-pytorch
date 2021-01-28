from typing import Optional
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


class ScheduledMixin:
    def __init__(self):
        object.__setattr__(self, 'schedules', dict())
        object.__setattr__(self, '_scheduled_objects', dict())

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in self.schedules:
            self.schedules.pop(name)
        if isinstance(value, Schedule):
            self.schedules[name] = value
        if isinstance(value, ScheduledMixin) and name in self._scheduled_objects:
            self._scheduled_objects[name] = value
        if name == 'global_t':
            for s in self._scheduled_objects.values():
                s.global_t = value

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
        if name in self._scheduled_objects:
            self._scheduled_objects.pop(name)

    def raw_attribute(self, name):
        if name in self.schedules:
            return self.schedules[name]
        return getattr(self, name)


class Trainer(ScheduledMixin):
    DEFAULT_NAME = 'default'

    def __init__(self, model_fn,
                 learning_rate: float,
                 loggers=None,
                 project: str = None,
                 name: str = None,
                 max_time_steps: Optional[int] = None,
                 save_interval: int = 50000,
                 log_interval: int = 5000,
                 num_gpus: Optional[int] = None,
                 max_episodes: Optional[int] = None):
        super().__init__()
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
            _, episode_lengths, returns = self.collect_experience()
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

            num_steps, output = self.training_step(batch)
            if hasattr(self, 'on_training_step_end'):
                self.on_training_step_end(batch, output)
            self.global_t += num_steps * self.world_size
            progress.update(num_steps * self.world_size)

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
                metrics = self._collect_logs()
                self._last_log_step = self.global_t
                postfix = 'return: {return:.2f}'.format(**metrics)
                if 'loss' in metrics:
                    postfix += ', loss: {loss:.4f}'.format(**metrics)
                progress.set_postfix_str(postfix)

        self.save()
        progress.close()

    def evaluate(self, max_time_steps=None, max_episodes=None):
        assert max_time_steps is not None or max_episodes is not None
        self._setup('evaluate')
        progress = tqdm.tqdm(desc='evaluating', total=max_time_steps or max_episodes)
        metrics = self._get_metrics_context()
        local_frames = 0
        local_t = 0
        tstart = time.time()
        max_t = max_time_steps or max_episodes
        last_progress_step = 0
        while True:
            diff, episode_lengths, returns = self.collect_experience()
            metrics['return'](returns)
            metrics['episode_length'](episode_lengths)

            # We multiply by the world size, since the reduction in distributed setting is mean
            metrics['episodes'](len(returns) * self.world_size)
            metrics['updates'](1)
            metrics['fps'](local_frames / (time.time() - tstart))
            local_frames += diff
            if max_episodes is not None:
                diff = len(returns)
            progress.update(diff)
            local_t += diff
            if local_t >= max_t:
                break

            if local_frames - last_progress_step >= self.log_interval:
                mval = {k: v() for k, v in metrics.items()}
                last_progress_step = local_frames
                postfix = 'return: {return:.2f}'.format(**mval)
                if 'loss' in mval:
                    postfix += ', loss: {loss:.4f}'.format(**mval)
                progress.set_postfix_str(postfix)
        progress.close()
        return metrics.collect()

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

    @staticmethod
    def _get_metrics_context():
        return metrics.MetricsContext(**{
            'return': metrics.Mean(),
            'updates': metrics.AccumulatedMetric(lambda x, y: x + y),
            'fps': metrics.LastValue(),
            'episodes': metrics.Mean(is_distributed=True)})

    def _setup(self, stage):
        # Setup schedules
        for schedule in self.schedules.values():
            if hasattr(schedule, 'total_iterations') and schedule.total_iterations is None:
                if self.max_time_steps is not None:
                    schedule.total_iterations = self.max_time_steps
                else:
                    raise RuntimeError(f'Schedule {schedule} does not have the max_time_steps set up.')

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
            self.metrics = self._get_metrics_context()
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

    def collect_experience(self):
        raise NotImplementedError()

    def sample_experience_batch(self):
        raise NotImplementedError()

    def configure_optimizers(self, model):
        raise NotImplementedError()

    def training_step(self):
        raise NotImplementedError()

    def state_dict(self):
        state_dict = dict()

        def serialize_value(obj):
            if hasattr(obj, 'state_dict'):
                obj = obj.state_dict()
            if isinstance(obj, tuple):
                return tuple(map(serialize_value, obj))
            elif isinstance(obj, list):
                return list(map(serialize_value, obj))
            elif isinstance(obj, dict):
                return {k: serialize_value(v) for k, v in obj.items()}
            return obj

        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            if k in ['schedules', 'metrics', 'model_fn', 'env_fn', '_scheduled_objects']:
                continue
            if k in self.schedules:
                v = self.schedules[k]
            state_dict[k] = v
        return serialize_value(state_dict)

    def load_state_dict(self, state_dict):
        def deserialize_value(obj):
            if isinstance(obj, dict):
                if '_schedule_class' in obj:
                    return Schedule.from_state_dict(obj)
                return {k: deserialize_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return list(map(deserialize_value, obj))
            elif isinstance(obj, tuple):
                return tuple(map(deserialize_value, obj))
            return obj
        for k, v in state_dict.items():
            if hasattr(self, k) and hasattr(getattr(self, k), 'load_state_dict') and (not isinstance(v, dict) or '_schedule_class' not in v):
                getattr(self, k).load_state_dict(v)
            else:
                setattr(self, k, deserialize_value(v))
        return self
