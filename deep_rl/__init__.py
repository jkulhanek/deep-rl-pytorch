from .registry import make_agent, make_trainer, register_agent, register_trainer
from .configuration import configuration

VERSION = '0.2.1'

def configure(**kwargs):
    configuration.update(kwargs)