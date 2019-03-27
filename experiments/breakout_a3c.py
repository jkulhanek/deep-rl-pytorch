import gym
import numpy as np

from deep_rl import register_trainer
from deep_rl.a3c import A3CTrainer
from deep_rl.a2c.model import Flatten
from deep_rl.common.env import ScaledFloatFrame, TransposeImage, RewardCollector
from baselines.common.atari_wrappers import ClipRewardEnv, WarpFrame, NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FrameStack
from torch import nn

class CNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        def init_layer(layer, activation = None, gain = None):
            if activation is not None and gain is None:
                gain = nn.init.calculate_gain(activation.lower())
            elif activation is None and gain is None:
                gain = 1.0

            nn.init.orthogonal_(layer.weight.data, gain = gain)
            nn.init.zeros_(layer.bias.data)
            output = [layer]
            if activation is not None:
                output.append(getattr(nn, activation)())
            return output

        layers = []
        layers.extend(init_layer(nn.Conv2d(num_inputs, 32, 8, stride = 4), activation='ReLU'))
        layers.extend(init_layer(nn.Conv2d(32, 64, 4, stride = 2), activation='ReLU'))
        layers.extend(init_layer(nn.Conv2d(64, 32, 3, stride = 1), activation='ReLU'))
        layers.append(Flatten())
        layers.extend(init_layer(nn.Linear(32 * 7 * 7, 512), activation='ReLU'))
        
        self.main = nn.Sequential(*layers)
        self.critic = init_layer(nn.Linear(512, 1))[0]
        self.policy_logits = init_layer(nn.Linear(512, num_outputs), gain = 0.01)[0]

    def forward(self, inputs, states):
        main_features = self.main(inputs)
        policy_logits = self.policy_logits(main_features)
        critic = self.critic(main_features)
        return policy_logits, critic, states

    @property
    def output_names(self):
        return ('policy_logits', 'value', 'states')

class ConvertToNumpy(gym.ObservationWrapper):
    def observation(self, observation):
        return np.array(observation)

@register_trainer(max_time_steps = 10e6, validation_period = None,  episode_log_interval = 10, save = False)
class Trainer(A3CTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 16
        self.num_steps = 20
        self.gamma = .99

    def create_env(self, env):
        env = gym.make(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = RewardCollector(env)
        env = EpisodicLifeEnv(env)
        env = ClipRewardEnv(env)
        env = WarpFrame(env)
        env = FrameStack(env, 4)
        env = ConvertToNumpy(env)
        env = TransposeImage(env)
        env = ScaledFloatFrame(env)
        return env

    def create_model(self):
        return CNN(4, 4)

def default_args():
    return dict(
        env_kwargs = 'BreakoutNoFrameskip-v4',
        model_kwargs = dict()
    )