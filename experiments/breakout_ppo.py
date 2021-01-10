from torch import nn
import torch
from deep_rl.common.pytorch import forward_masked_rnn_transposed
import gym

from deep_rl import register_trainer
from deep_rl.model import TimeDistributed, Flatten
from deep_rl.actor_critic import PPO
from deep_rl.actor_critic.environment import create_unreal_env
from gym.vector import SyncVectorEnv, AsyncVectorEnv
# from baselines.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, ClipRewardEnv, WarpFrame, ScaledFloatFrame
from deep_rl.common.env import RewardCollector, TransposeImage


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
            nn.Conv2d(num_inputs, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
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
        self.rnn = nn.LSTM(256,
                           hidden_size=self.lstm_hidden_size,
                           num_layers=self.lstm_layers,
                           batch_first=True)

        self.apply(self.init_weights)

    def initial_states(self, batch_size):
        return tuple([torch.zeros([batch_size, self.lstm_layers, self.lstm_hidden_size], dtype=torch.float32) for _ in range(2)])

    def forward(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        policy_logits = self.policy_logits(features)
        critic = self.critic(features)
        return [policy_logits, critic, states]

    def _forward_base(self, observations, masks, states):
        shape = list(observations.shape)
        shape.insert(-2, 1)
        observations = observations.view(*shape)
        features = self.conv_base(observations)
        features = self.conv_merge(features)
        return forward_masked_rnn_transposed(features, masks, states, self.rnn.forward)


@register_trainer(max_time_steps=10e6, validation_period=None,  episode_log_interval=10, save=False)
class Trainer(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 16
        self.num_steps = 128
        self.gamma = .99

    def create_env(self, env):
        def wrap(env): return RewardCollector(gym.wrappers.AtariPreprocessing(env, terminal_on_life_loss=True))
        venv = gym.vector.AsyncVectorEnv([lambda: wrap(gym.make(**env))] * self.num_processes)
        self.validation_env = gym.vector.SyncVectorEnv([lambda: wrap(gym.make(**env))])
        return venv

    def create_model(self):
        return LSTMModel(1, self.env.single_action_space.n)


def default_args():
    return dict(
        env_kwargs=dict(id='BreakoutNoFrameskip-v4'),
        model_kwargs=dict()
    )
