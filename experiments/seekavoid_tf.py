import environments
import numpy as np
import gym

import torch.nn as nn
import torch

from deep_rl import register_trainer

from deep_rl.unreal_tensorflow2.main import A3CTrainer
from deep_rl.common.env import ScaledFloatFrame, RewardCollector, TransposeImage
from deep_rl.common.vec_env import SubprocVecEnv, DummyVecEnv

@register_trainer(max_time_steps = 40e6, validation_period = 50000, validation_episodes = 20,  episode_log_interval = 10, saving_period = 500000, save = True)
class T(A3CTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def default_args():
    return dict(
        env_kwargs = dict(id = 'DeepmindLabSeekavoidArena01-v0'),
        model_kwargs = dict()
    )