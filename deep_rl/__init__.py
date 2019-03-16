from .registry import make_agent, make_trainer, register_agent, register_trainer
from .configuration import configuration

def configure(**kwargs):
    configuration.update(kwargs)