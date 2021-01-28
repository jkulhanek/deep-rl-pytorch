import gym
import argparse
import numpy as np
from deep_rl import actor_critic
from deep_rl.utils.model import TimeDistributed
from deep_rl.utils.argparse import add_arguments
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.layer = TimeDistributed(nn.Linear(4, 256))
        self.action = TimeDistributed(nn.Linear(256, 2))
        self.critic = TimeDistributed(nn.Linear(256, 1))
        self.apply(init_weights)

    def forward(self, inputs, masks, states):
        features = self.layer(inputs)
        features = F.relu(features)
        value = self.critic(features)
        policy_logits = self.action(features)
        return policy_logits, value, states


ALGO_MAP = {k.lower(): getattr(actor_critic, k) for k in dir(actor_critic) if k[0].isupper() and 'unreal' not in k.lower()}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--trainer', choices=sorted(ALGO_MAP.keys()), default='ppo')
    parser.add_argument('--env', default='CartPole-v0')
    parser.add_argument('--num-eval-episodes', type=int, default=50)
    args, _ = parser.parse_known_args()
    trainer_class = ALGO_MAP[args.trainer]

    parser, bind_arguments = add_arguments(parser, trainer_class, defaults=dict(max_time_steps=60000, log_interval=1000, save_interval=10000), loggers=[], project='deep-rl')
    parser = argparse.ArgumentParser(parents=[parser], add_help=True)
    args = parser.parse_args()

    def env_fn(rank):
        env = gym.make(args.env)
        env = gym.wrappers.TransformObservation(env, lambda x: x.astype(np.float32))
        return env

    trainer = trainer_class(Model, env_fn, **bind_arguments(args))

    trainer.fit()
    return trainer.evaluate(max_episodes=args.num_eval_episodes)


if __name__ == '__main__':
    main()
