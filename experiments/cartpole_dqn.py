import gym
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import deep_rl.deepq as deepq
from deep_rl import register_trainer


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.layer = nn.Linear(4, 256)
        self.adventage = nn.Linear(256, 2)
        self.value = nn.Linear(256, 1)
        self.apply(init_weights)

    def forward(self, inputs):
        features = self.layer(inputs)
        features = F.relu(features)
        value = self.value(features)
        adventage = self.adventage(features)
        features = adventage + value - adventage.mean()
        return features


@register_trainer(max_time_steps=100000, episode_log_interval=10)
class Trainer(deepq.DeepQTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.annealing_steps = 10000
        self.preprocess_steps = 1000
        self.replay_size = 50000
        self.allow_gpu = False

    def create_model(self):
        return Model()

    def create_env(self, env):
        env = super().create_env(env)

        class W(gym.ObservationWrapper):
            def observation(self, o):
                return o.astype(np.float32)
        return W(env)


def default_args():
    return dict(
        env_kwargs=dict(id='CartPole-v0'),
        model_kwargs=dict()
    )
