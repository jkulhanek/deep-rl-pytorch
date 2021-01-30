import gym
import numpy as np
import argparse
from functools import partial
from deep_rl.utils.environment import TransposeWrapper
from deep_rl import deepq
from deep_rl.utils.argparse import add_arguments
from deep_rl.deepq.models import RainbowConvModel

ALGO_MAP = {k.lower(): getattr(deepq, k) for k in dir(deepq) if k[0].isupper()}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--trainer', choices=sorted(ALGO_MAP.keys()), default='rainbow')
    parser.add_argument('--env', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--num-eval-episodes', type=int, default=50)
    parser.add_argument('--frame-stack', type=int, default=4)
    args, _ = parser.parse_known_args()
    trainer_class = ALGO_MAP[args.trainer]

    parser, bind_arguments = add_arguments(parser, trainer_class, defaults=dict(max_time_steps=10000), loggers=[], project='deep-rl')
    parser = argparse.ArgumentParser(parents=[parser], add_help=True)
    args = parser.parse_args()

    def env_fn():
        env = gym.wrappers.FrameStack(gym.wrappers.AtariPreprocessing(gym.make(args.env), terminal_on_life_loss=True, scale_obs=True, grayscale_newaxis=True), args.frame_stack)
        env = gym.wrappers.TransformObservation(env, lambda x: np.array(x))
        env = TransposeWrapper(env)
        return env

    if 'rainbow' not in args.trainer:
        model_fn = partial(RainbowConvModel, 4, distributional=False, noisy_nets=False)
    else:
        model_fn = partial(RainbowConvModel, 4, distributional=args.distributional,
                           noisy_nets=args.noisy_nets, distributional_atoms=args.distributional_atoms)

    trainer = trainer_class(model_fn, env_fn, **bind_arguments(args))

    trainer.fit()
    return trainer.evaluate(max_episodes=args.num_eval_episodes)


if __name__ == '__main__':
    main()
