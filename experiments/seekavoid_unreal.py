import environments
import numpy as np

from deep_rl import register_trainer
from deep_rl.unreal import UnrealTrainer
from deep_rl.a2c.model import TimeDistributedConv

@register_trainer(max_time_steps = 40e6, validation_period = 50000, validation_episodes = 20,  episode_log_interval = 10, saving_period = 500000, save = True)
class Trainer(UnrealTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 8
        self.max_gradient_norm = 40.0
        self.rms_alpha = 0.99
        self.rms_epsilon = 0.1
        self.num_steps = 20
        self.gamma = .99

    def create_model(self):
        return TimeDistributedConv(self.env.observation_space.shape[0], self.env.action_space.n)

def default_args():
    return dict(
        env_kwargs = 'DeepmindLabSeekavoidArena01-v0',
        model_kwargs = dict()
    )