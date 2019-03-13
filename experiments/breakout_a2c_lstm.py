from torch import nn
import torch
import math
from deep_rl.a2c.model import TimeDistributed, Flatten
from deep_rl.common.pytorch import forward_masked_rnn_transposed
import gym
import numpy as np

from deep_rl import register_trainer
from deep_rl.a2c import A2CTrainer
from deep_rl.common.vec_env import DummyVecEnv, SubprocVecEnv
from baselines.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, ClipRewardEnv, WarpFrame, ScaledFloatFrame
from deep_rl.common.env import VecFrameStack, VecTransposeImage, RewardCollector, TransposeImage
from deep_rl.a2c_unreal.util import UnrealEnvBaseWrapper

class LSTMModel(nn.Module):
    def init_weights(self, module):
        if type(module) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

        elif type(module) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
            nn.init.zeros_(module.bias.data)
            nn.init.xavier_uniform_(module.weight.data)

    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.conv_base = TimeDistributed(nn.Sequential(
            nn.Conv2d(num_inputs, 16, 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride = 2),
            nn.ReLU(),
        ))

        self.conv_merge = TimeDistributed(nn.Sequential(
            Flatten(),
            nn.Linear(9 ** 2 * 32, 256),
            nn.ReLU()
        ))

        self.main_output_size = 256
        
        self.critic = TimeDistributed(nn.Linear(self.main_output_size, 1))
        self.policy_logits = TimeDistributed(nn.Linear(self.main_output_size, num_outputs))

        self.lstm_layers = 1
        self.lstm_hidden_size = 256
        self.rnn = nn.LSTM(256 + num_outputs + 1,
            hidden_size = self.lstm_hidden_size, 
            num_layers = self.lstm_layers,
            batch_first = True)

        self.apply(self.init_weights)

    def initial_states(self, batch_size):
        return tuple([torch.zeros([batch_size, self.lstm_layers, self.lstm_hidden_size], dtype = torch.float32) for _ in range(2)])

    def forward(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        policy_logits = self.policy_logits(features)
        critic = self.critic(features)
        return [policy_logits, critic, states]

    def _forward_base(self, inputs, masks, states):
        observations, last_reward_action = inputs
        features = self.conv_base(observations)
        features = self.conv_merge(features)
        features = torch.cat((features, last_reward_action,), dim = 2)
        return forward_masked_rnn_transposed(features, masks, states, self.rnn.forward)

@register_trainer(max_time_steps = 10e6, validation_period = None,  episode_log_interval = 10, save = False)
class Trainer(A2CTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 16
        self.num_steps = 20
        self.gamma = .99

    def create_env(self, env):
        env_base = env
        def _thunk():
            env = gym.make(**env_base)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = RewardCollector(env)
            env = EpisodicLifeEnv(env)
            env = ClipRewardEnv(env)
            env = WarpFrame(env)
            env = ScaledFloatFrame(env)
            env = TransposeImage(env)
            env = UnrealEnvBaseWrapper(env)
            return env

        self.validation_env = DummyVecEnv([_thunk])
        return SubprocVecEnv([_thunk for _ in range(self.num_processes)])

    def create_model(self):
        return LSTMModel(self.env.observation_space.shape[0], self.env.action_space.n)

def default_args():
    return dict(
        env_kwargs = dict(id = 'BreakoutNoFrameskip-v4'),
        model_kwargs = dict()
    )