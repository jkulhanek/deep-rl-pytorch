import environments
import numpy as np
import gym

from deep_rl import register_trainer
from deep_rl.a2c import A2CTrainer
from deep_rl.common.schedules import LinearSchedule
from deep_rl.a2c.model import TimeDistributed, Flatten, TimeDistributedModel
from deep_rl.common.pytorch import forward_masked_rnn_transposed
from deep_rl.common.env import ScaledFloatFrame, RewardCollector, TransposeImage
from deep_rl.common.vec_env import SubprocVecEnv, DummyVecEnv

from torch import nn
import torch


class Model(TimeDistributedModel):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        layers = []
        layers.extend(self.init_layer(nn.Conv2d(num_inputs, 16, 8, stride = 4), activation='ReLU'))
        layers.extend(self.init_layer(nn.Conv2d(16, 32, 4, stride = 2), activation='ReLU'))
        layers.append(TimeDistributed(Flatten()))
        layers.extend(self.init_layer(nn.Linear(9 ** 2 * 32, 256), activation='ReLU'))
        self.conv = nn.Sequential(*layers)
        self.main_output_size = 256
        
        self.critic = self.init_layer(nn.Linear(self.main_output_size, 1))[0]
        self.policy_logits = self.init_layer(nn.Linear(self.main_output_size, num_outputs), gain = 0.01)[0]

        self.lstm_layers = 1
        self.lstm_hidden_size = 256
        self.rnn = nn.LSTM(256,
            hidden_size = self.lstm_hidden_size, 
            num_layers = self.lstm_layers,
            batch_first = True)

    def initial_states(self, batch_size):
        return tuple([torch.zeros([batch_size, self.lstm_layers, self.lstm_hidden_size], dtype = torch.float32) for _ in range(2)])

    def forward(self, inputs, masks, states):
        observations = inputs
        features = self.conv(observations)
        features, states = forward_masked_rnn_transposed(features, masks, states, self.rnn.forward)
        policy_logits = self.policy_logits(features)
        critic = self.critic(features)
        return [policy_logits, critic, states]

@register_trainer(max_time_steps = 40e6, validation_period = 50000, validation_episodes = 20,  episode_log_interval = 10, saving_period = 500000, save = True)
class Trainer(A2CTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 16
        self.max_gradient_norm = 0.5
        self.rms_alpha = 0.99
        self.rms_epsilon = 1e-5
        self.num_steps = 20
        self.gamma = .99
        self.allow_gpu = True
        self.learning_rate = LinearSchedule(7e-4, 0, self.max_time_steps)

    def create_model(self):
        return Model(self.env.observation_space.shape[0], self.env.action_space.n)

    def create_env(self, env):
        def thunk(env):
            env = gym.make(**env)
            env = RewardCollector(env)
            env = TransposeImage(env)
            env = ScaledFloatFrame(env)
            return env
        
        self.validation_env = DummyVecEnv([lambda: thunk(env)])
        env = SubprocVecEnv([lambda: thunk(env) for _ in range(self.num_processes)])
        return env

def default_args():
    return dict(
        env_kwargs = dict(id = 'DeepmindLabNavMazeStatic01-v0'),
        model_kwargs = dict()
    )