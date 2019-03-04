import gym
import numpy as np

from deep_rl import register_trainer
from deep_rl.a2c import A2CTrainer
from deep_rl.a2c.model import TimeDistributedConv

@register_trainer(max_time_steps = 10e6, validation_period = None,  episode_log_interval = 10, save = False)
class Trainer(A2CTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 16
        self.num_steps = 5
        self.gamma = .99

    def create_model(self):
        return TimeDistributedConv(self.env.observation_space.shape[0], self.env.action_space.n)

def default_args():
    return dict(
        env_kwargs = 'BreakoutNoFrameskip-v4',
        model_kwargs = dict()
    )