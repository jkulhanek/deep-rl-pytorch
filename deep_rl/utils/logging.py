from abc import abstractmethod, ABC
import os
import sys
import logging
from typing import Optional, Union, Dict, Any
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from torch.utils.tensorboard.summary import hparams
from collections import defaultdict
try:
    import wandb
    from wandb.wandb_run import Run
except ImportError:  # pragma: no-cover
    wandb = None
    Run = None


def rank_zero_only(fn):
    def _fn(*args, **kwargs):
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)
    return _fn


setattr(rank_zero_only, 'rank', 0)


def rank_zero_experiment(fn):
    def _fn(*args, **kwargs):
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)

        class FakeExperiment:
            def __getattr__(self, name):
                return lambda *args, **kwargs: None

        return FakeExperiment()
    return _fn


def save_hparams_to_yaml(config_yaml, hparams: Union[dict, Namespace]) -> None:
    """
    Args:
        config_yaml: path to new YAML file
        hparams: parameters to be saved
    """
    if not os.isdir(os.path.dirname(config_yaml)):
        raise RuntimeError(f"Missing folder: {os.path.dirname(config_yaml)}.")

    # convert Namespace or AD to dict
    if isinstance(hparams, Namespace):
        hparams = vars(hparams)
    elif type(hparams).__name__ == 'AttributeDict':
        hparams = dict(hparams)


class LoggerBase(ABC):
    def __init__(self):
        super().__init__()
        self.project = None
        self.experiment_name = None

    def setup(self, phase, trainer):
        self.project = trainer.project
        self.experiment_name = trainer.experiment_name

    @abstractmethod
    def log_metrics(self, metrics, step):
        pass

    @property
    @abstractmethod
    def experiment(self):
        pass

    def save(self) -> None:
        pass

    def finalize(self) -> None:
        pass


class ExperimentTuple:
    def __init__(self, wandb_experiment: Run, tensorboard_experiment: SummaryWriter):
        self.wandb_experiment = wandb_experiment
        self.tensorboard_experiment = tensorboard_experiment
        self._wandb_offset = 0

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return (self.wandb_experiment, self.tensorboard_experiment)[idx]

    def __getattr__(self, name):
        return getattr(self.tensorboard_experiment, name)

    def add_image(self, tag, img_tensor, global_step=None, *, label=None, **kwargs):
        self.tensorboard_experiment.add_image(tag, img_tensor, global_step=global_step, **kwargs)
        self.wandb_experiment.log({tag: [wandb.Image(img_tensor, caption=label or tag)]}, step=global_step + self._wandb_offset if global_step is not None else global_step)

    def add_images(self, tag, img_tensor, global_step=None, *, label=None, **kwargs):
        self.tensorboard_experiment.add_images(tag, img_tensor, global_step=global_step, **kwargs)
        self.wandb_experiment.log({tag: [wandb.Image(x, caption=f'{label or tag} {i}') for i, x in torch.unbind(img_tensor)]},
                                  step=global_step + self._wandb_offset if global_step is not None else global_step)


class WandbLogger(LoggerBase):
    LOGGER_JOIN_CHAR = '-'
    NAME_HPARAMS_FILE = 'hparams.yaml'

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: bool = False,
        id: Optional[str] = None,
        anonymous: bool = False,
        version: Optional[str] = None,
        project: Optional[str] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        experiment=None,
        prefix: str = '',
        **kwargs
    ):
        if wandb is None:
            raise ImportError('You want to use `wandb` logger which is not installed yet,'  # pragma: no-cover
                              ' install it with `pip install wandb`.')
        super().__init__()
        self.experiment_name = name
        self._save_dir = save_dir
        self._anonymous = 'allow' if anonymous else None
        self._id = version or id
        self.project = project
        self._experiment = experiment
        self._offline = offline
        self._prefix = prefix
        self._log_graph = log_graph
        self._default_hp_metric = default_hp_metric
        self._kwargs = kwargs
        # logging multiple Trainer on a single W&B run (k-fold, resuming, etc)
        self._step_offset = 0
        self.hparams = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        # args needed to reload correct experiment
        state['_id'] = self._experiment.wandb_experiment.id if self._experiment is not None else None

        # cannot be pickled
        state['_experiment'] = None
        return state

    @property
    @rank_zero_experiment
    def experiment(self) -> Run:
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            if self._offline:
                os.environ['WANDB_MODE'] = 'dryrun'
            wandb_experiment = wandb.init(
                name=self.experiment_name, dir=self._save_dir, project=self.project, anonymous=self._anonymous,
                id=self._id, resume='allow', **self._kwargs) if wandb.run is None else wandb.run

            # offset logging step when resuming a run
            self._step_offset = wandb_experiment.step
            # save checkpoints in wandb dir to upload on W&B servers
            self._save_dir = wandb_experiment.dir
            tensorboard_experiment = SummaryWriter(log_dir=wandb_experiment.dir, **self._kwargs)
            self._experiment = ExperimentTuple(wandb_experiment, tensorboard_experiment)
            self._experiment._wandb_offset = self._step_offset
        return self._experiment

    def setup(self, phase, trainer):
        if self.experiment_name is None:
            self.experiment_name = trainer.name
        if self.project is None:
            self.project = trainer.project

        if phase == 'fit':
            # Setup experiment to get the save path
            _ = self.experiment

    @rank_zero_only
    def watch(self, model: nn.Module, log: str = 'gradients', log_freq: int = 100):
        wandb_experiment, tensorboard_experiment = self.experiment
        wandb_experiment.watch(model, log=log, log_freq=log_freq)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace],
                        metrics: Optional[Dict[str, Any]] = None) -> None:
        params = self._convert_params(params)
        wandb_experiment, tensorboard_experiment = self.experiment

        # store params to output
        self.hparams.update(params)
        params = self._flatten_dict(params)
        params = self._sanitize_callable_params(params)
        if metrics is None:
            if self._default_hp_metric:
                metrics = {"hp_metric": -1}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        # TensorBoard
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                tensorboard_experiment.add_scalar(k, v, 0)
            exp, ssi, sei = hparams(params, metrics)
            writer = tensorboard_experiment._get_file_writer()
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)

        # Wandb
        wandb_experiment.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'
        wandb_experiment, tensorboard_experiment = self.experiment

        # TensorBoard
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                tensorboard_experiment.add_scalars(k, v, step)
            else:
                try:
                    tensorboard_experiment.add_scalar(k, v, step)
                except Exception as e:
                    m = f'\n you tried to log {v} which is not currently supported. Try a dict or a scalar/tensor.'
                    type(e)(e.message + m)

        # Wandb
        if step is not None and step + self._step_offset < wandb_experiment.step:
            logging.warn('Trying to log at a previous step. Use `commit=False` when logging metrics manually.')
        wandb_experiment.log(metrics, step=(step + self._step_offset) if step is not None else None)

    @rank_zero_only
    def log_graph(self, model, input_array=None):
        if self._log_graph:
            wandb_experiment, tensorboard_experiment = self.experiment
            if input_array is None:
                input_array = model.example_input_array

            if input_array is not None:
                input_array = model.transfer_batch_to_device(input_array, model.device)
                tensorboard_experiment.add_graph(model, input_array)
            else:
                logging.warn('Could not log computational graph since the'
                             ' `model.example_input_array` attribute is not set'
                             ' or `input_array` was not given',
                             UserWarning)

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def name(self) -> Optional[str]:
        # don't create an experiment if we don't have one
        return self._experiment.wandb_experiment.project_name() if self._experiment else self._name

    @property
    def version(self) -> Optional[str]:
        # don't create an experiment if we don't have one
        return self._experiment.wandb_experiment.id if self._experiment else self._id

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.experiment.flush()
        self.save()

        # offset future training logged on same W&B run
        if self._experiment is not None:
            self._step_offset = self._experiment.wandb_experiment.step

        # upload all checkpoints from saving dir
        wandb.save(os.path.join(self.save_dir, "*.pth"))
        wandb.save(os.path.join(self.save_dir, self.NAME_HPARAMS_FILE))

    @rank_zero_only
    def save(self) -> None:
        # Initialize experiment
        _ = self.experiment

        super().save()

        # prepare the file path
        hparams_file = os.path.join(self.save_dir, self.NAME_HPARAMS_FILE)

        # save the metatags file if it doesn't exist
        if not os.path.isfile(hparams_file):
            save_hparams_to_yaml(hparams_file, self.hparams)


class TensorBoardLogger(LoggerBase):
    LOGGER_JOIN_CHAR = '-'
    NAME_HPARAMS_FILE = 'hparams.yaml'

    def __init__(
        self,
        log_graph: bool = False,
        **kwargs
    ):
        super().__init__()
        self._log_graph = log_graph
        self._kwargs = kwargs
        self._experiment = None
        self.hparams = {}
        self.save_dir = None

    def __getstate__(self):
        state = self.__dict__.copy()

        # cannot be pickled
        state['_experiment'] = None
        return state

    @property
    @rank_zero_experiment
    def experiment(self) -> Run:
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            assert self.save_dir is not None
            tensorboard_experiment = SummaryWriter(log_dir=self.save_dir, **self._kwargs)
            self._experiment = tensorboard_experiment
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace],
                        metrics: Optional[Dict[str, Any]] = None) -> None:
        params = self._convert_params(params)

        # store params to output
        self.hparams.update(params)
        params = self._flatten_dict(params)
        params = self._sanitize_callable_params(params)
        if metrics is None:
            if self._default_hp_metric:
                metrics = {"hp_metric": -1}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        # TensorBoard
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.experiment.add_scalar(k, v, 0)
            exp, ssi, sei = hparams(params, metrics)
            writer = self.experiment._get_file_writer()
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        # TensorBoard
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                self.experiment.add_scalars(k, v, step)
            else:
                try:
                    self.experiment.add_scalar(k, v, step)
                except Exception as e:
                    m = f'\n you tried to log {v} which is not currently supported. Try a dict or a scalar/tensor.'
                    type(e)(e.message + m)

    @rank_zero_only
    def log_graph(self, model, input_array=None):
        if self._log_graph:
            if input_array is None:
                input_array = model.example_input_array

            if input_array is not None:
                input_array = model.transfer_batch_to_device(input_array, model.device)
                self.experiment.add_graph(model, input_array)
            else:
                logging.warn('Could not log computational graph since the'
                             ' `model.example_input_array` attribute is not set'
                             ' or `input_array` was not given',
                             UserWarning)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.experiment.flush()
        self.save()

    @rank_zero_only
    def save(self) -> None:
        # Initialize experiment
        _ = self.experiment

        super().save()

        # prepare the file path
        hparams_file = os.path.join(self.save_dir, self.NAME_HPARAMS_FILE)

        # save the metatags file if it doesn't exist
        if not os.path.isfile(hparams_file):
            save_hparams_to_yaml(hparams_file, self.hparams)


class CsvLogger(LoggerBase):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self._experiment = None
        self.save_dir = None
        self._metrics = defaultdict(lambda: ([], []))
        self._was_initialized = False

    def __getstate__(self):
        state = self.__dict__.copy()

        # cannot be pickled
        state['_experiment'] = None
        return state

    @property
    @rank_zero_experiment
    def experiment(self) -> Run:
        return self._experiment

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        for (tag, val) in metrics.items():
            t, v = self._metrics[tag]
            t.append(step)
            v.append(val)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()

    @rank_zero_only
    def save(self) -> None:
        import csv
        with open(os.path.join(self.save_dir, 'metrics.txt'), 'a' if self._was_initialized else 'w+') as f:
            writer = csv.writer(f)
            for key, vals in self._metrics.items():
                writer.writerow([key, str(len(vals[0]))] + vals[0] + vals[1])

            f.flush()
        self._was_initialized = True
        self._metrics = defaultdict(lambda: ([], []))


def setup_logging(level=logging.INFO):
    from tqdm import tqdm

    def is_console_handler(handler):
        return isinstance(handler, logging.StreamHandler) and handler.stream in {sys.stdout, sys.stderr}

    class TqdmLoggingHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:  # noqa pylint: disable=bare-except
                self.handleError(record)

    logging.basicConfig(stream=sys.stdout, level=level)
    handler = TqdmLoggingHandler(sys.stdout)
    try:
        import colorlog
        formatter = colorlog.LevelFormatter(fmt={
            'DEBUG': '%(log_color)sdebug: %(message)s (%(module)s:%(lineno)d)%(reset)s',
            'INFO': '%(log_color)sinfo%(reset)s: %(message)s',
            'WARNING': '%(log_color)swarning%(reset)s: %(message)s (%(module)s:%(lineno)d)',
            'ERROR': '%(log_color)serror%(reset)s: %(message)s (%(module)s:%(lineno)d)',
            'CRITICAL': '%(log_color)scritical: %(message)s (%(module)s:%(lineno)d)%(reset)s',
        }, log_colors={
            'DEBUG': 'white',
            'INFO': 'bold_green',
            'WARNING': 'bold_yellow',
            'ERROR': 'bold_red',
            'CRITICAL': 'bold_red',
        })
        handler.setFormatter(formatter)
    except(ModuleNotFoundError):
        # We do not require colorlog to be present
        pass
    logging._acquireLock()
    orig_handlers = logging.root.handlers
    try:
        logging.root.handlers = [x for x in orig_handlers if not is_console_handler(x)] + [handler]
    except Exception:
        logging.root.handlers = orig_handlers
    finally:
        logging._releaseLock()
