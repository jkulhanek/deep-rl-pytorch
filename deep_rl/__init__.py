from .registry import make_agent, make_trainer, register_agent, register_trainer
from .configuration import configuration
from .common import schedules

VERSION = '0.3.0'


def configure(**kwargs):
    configuration.update(kwargs)
