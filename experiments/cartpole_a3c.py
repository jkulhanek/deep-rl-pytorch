import gym
import numpy as np

from deep_rl import register_trainer
from deep_rl.actor_critic import A3C
from deep_rl.common.env import RewardCollector
from gym.vector import AsyncVectorEnv, SyncVectorEnv
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.layer = nn.Linear(4, 256)
        self.action = nn.Linear(256, 2)
        self.critic = nn.Linear(256, 1)
        self.apply(init_weights)

    def forward(self, inputs, states):
        features = self.layer(inputs)
        features = F.relu(features)
        value = self.critic(features)
        policy_logits = self.action(features)
        return policy_logits, value, states


@register_trainer(max_time_steps=10e6, validation_period=None,  episode_log_interval=10, save=False)
class Trainer(A3C):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 4
        self.num_steps = 20
        self.gamma = .99

        self.allow_gpu = False

    def create_env(self, env):
        class W(gym.ObservationWrapper):
            def observation(self, o):
                return o.astype(np.float32)

        env = gym.make(**env)
        env = RewardCollector(env)
        env = gym.wrappers.TransformReward(env, lambda r: 0.01 * r)
        env = W(env)
        return env

    def create_model(self):
        return Model()


def default_args():
    return dict(
        env_kwargs=dict(id='CartPole-v0'),
        model_kwargs=dict()
    )

