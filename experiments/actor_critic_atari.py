import gym
import argparse
from functools import partial
from deep_rl.actor_critic.model import UnrealModel
from deep_rl.utils.environment import TransposeWrapper
from deep_rl.utils import logging
from deep_rl import actor_critic
from deep_rl.utils.model import TimeDistributed, Flatten
from deep_rl.common.pytorch import forward_masked_rnn_transposed
import torch
from torch import nn


# from baselines.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, ClipRewardEnv, WarpFrame, ScaledFloatFrame


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

    def __init__(self, num_inputs, num_outputs, channels=[16, 32, 256]):
        super().__init__()
        assert len(channels) == 3

        self.conv_base = TimeDistributed(nn.Sequential(
            nn.Conv2d(num_inputs, channels[0], 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], 4, stride=2),
            nn.ReLU(),
        ))

        self.conv_merge = TimeDistributed(nn.Sequential(
            Flatten(),
            nn.Linear(9 ** 2 * channels[-2], channels[-1]),
            nn.ReLU()
        ))

        self.main_output_size = channels[-1]

        self.critic = TimeDistributed(nn.Linear(self.main_output_size, 1))
        self.policy_logits = TimeDistributed(nn.Linear(self.main_output_size, num_outputs))

        self.lstm_layers = 1
        self.lstm_hidden_size = channels[-1]
        self.rnn = nn.LSTM(channels[-1],
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
        features = self.conv_base(observations)
        features = self.conv_merge(features)
        return forward_masked_rnn_transposed(features, masks, states, self.rnn.forward)


ALGO_MAP = {k.lower(): getattr(actor_critic, k) for k in dir(actor_critic) if k[0].isupper()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer', choices=sorted(ALGO_MAP.keys()), default='ppo')
    parser.add_argument('--env', default='BreakoutNoFrameskip-v4')
    args = parser.parse_args()

    def env_fn(rank):
        env = gym.wrappers.AtariPreprocessing(gym.make(args.env), terminal_on_life_loss=True, scale_obs=True, grayscale_newaxis=True)
        env = TransposeWrapper(env)
        return env

    if 'unreal' in args.trainer:
        model_fn = partial(UnrealModel, 1, 4)
    else:
        model_fn = partial(LSTMModel, 1, 4)

    trainer = ALGO_MAP[args.trainer](model_fn, env_fn, project='deep-rl', loggers=[])
    trainer.fit()


if __name__ == '__main__':
    main()
