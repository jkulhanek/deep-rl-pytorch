import argparse
import gym
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from deep_rl import deepq
from deep_rl.utils.argparse import add_arguments


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


ALGO_MAP = {k.lower(): getattr(deepq, k) for k in dir(deepq) if k[0].isupper()}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--trainer', choices=sorted(ALGO_MAP.keys()), default='dqn')
    parser.add_argument('--env', default='CartPole-v0')
    parser.add_argument('--num-eval-episodes', type=int, default=50)
    args, _ = parser.parse_known_args()
    trainer_class = ALGO_MAP[args.trainer]

    parser, bind_arguments = add_arguments(parser, trainer_class, defaults=dict(max_time_steps=1000000, preload_steps=80000), loggers=[], project='deep-rl')
    parser = argparse.ArgumentParser(parents=[parser], add_help=True)
    args = parser.parse_args()

    def env_fn():
        env = gym.make(args.env)
        env = gym.wrappers.TransformObservation(env, lambda x: x.astype(np.float32))
        return env

    trainer = trainer_class(Model, env_fn, **bind_arguments(args))

    trainer.fit()
    return trainer.evaluate(max_episodes=args.num_eval_episodes)


if __name__ == '__main__':
    main()
