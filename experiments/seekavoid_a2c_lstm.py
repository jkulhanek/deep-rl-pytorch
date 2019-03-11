import environments
import numpy as np
import gym

import torch.nn as nn
import torch

from deep_rl import register_trainer
from deep_rl.a2c import A2CTrainer
from deep_rl.a2c.model import TimeDistributed, Flatten, TimeDistributedModel
from deep_rl.common.pytorch import forward_masked_rnn
from deep_rl.common.env import ScaledFloatFrame, RewardCollector, TransposeImage
from deep_rl.common.vec_env import SubprocVecEnv, DummyVecEnv

class UnrealEnvBaseWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_action_reward = None

    def reset(self):
        self.last_action_reward = np.zeros(self.action_space.n + 1, dtype = np.float32)
        return self.observation(self.env.reset())

    def step(self, action):
        observation, reward, done, stats = self.env.step(action)
        self.last_action_reward = np.zeros(self.action_space.n + 1, dtype = np.float32)
        self.last_action_reward[action] = 1.0
        self.last_action_reward[-1] = np.clip(reward, -1, 1)
        return self.observation(observation), reward, done, stats

    def observation(self, observation):
        return (observation, self.last_action_reward)

class UnrealModelBase(TimeDistributedModel):
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
        self.rnn = nn.LSTM(256 + num_outputs + 1, # Conv outputs + last action, reward
            hidden_size = self.lstm_hidden_size, 
            num_layers = self.lstm_layers,
            batch_first = True)

    def initial_states(self, batch_size):
        return tuple([torch.zeros([self.lstm_layers, batch_size, self.lstm_hidden_size], dtype = torch.float32) for _ in range(2)])

    def forward(self, inputs, masks, states):
        observations, last_reward_action = inputs
        conv_features = self.conv(observations)
        features = torch.cat((conv_features, last_reward_action,), dim = 2)
        features, states = forward_masked_rnn(features, masks, states, self.rnn.forward)
        policy_logits = self.policy_logits(features)
        critic = self.critic(features)
        return [policy_logits, critic, states]

    @property
    def output_names(self):
        return ('policy_logits', 'value', 'states')


@register_trainer(max_time_steps = 40e6, validation_period = 50000, validation_episodes = 20,  episode_log_interval = 10, saving_period = 500000, save = True)
class Trainer(A2CTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 8
        self.max_gradient_norm = 40.0
        self.rms_alpha = 0.99
        self.rms_epsilon = 0.1
        self.num_steps = 20
        self.gamma = .99
        self.data_parallel = False

    def create_model(self):
        return UnrealModelBase(self.env.observation_space.shape[0], self.env.action_space.n)

    def create_env(self, env):
        def thunk(env):
            env = gym.make(**env)
            env = RewardCollector(env)
            env = TransposeImage(env)
            env = ScaledFloatFrame(env)
            env = UnrealEnvBaseWrapper(env)
            return env
        
        self.validation_env = DummyVecEnv([lambda: thunk(env)])
        env = SubprocVecEnv([lambda: thunk(env) for _ in range(self.num_processes)])
        return env

def default_args():
    return dict(
        env_kwargs = dict(id = 'DeepmindLabSeekavoidArena01-v0'),
        model_kwargs = dict()
    )