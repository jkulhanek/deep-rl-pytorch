import tensorflow as tf
import threading
import numpy as np
from collections import namedtuple
import gym

import signal
import random
import math
import os
import time

from .environment.environment import Environment
from .model.model import UnrealModel
from .trainer import A3CAgent as Trainer
from .train.rmsprop_applier import RMSPropApplier

from ..common.multiprocessing import ThreadServerTrainer
from deep_rl.core import SingleTrainer

from deep_rl.common.env import ScaledFloatFrame, RewardCollector

from deep_rl import register_trainer, make_trainer

from experiments.seekavoid_a2c_lstm import UnrealEnvBaseWrapper

USE_GPU = True  # To use GPU, set True

# get command line args
flags = dict(env_type="lab",
             env_name="nav_maze_static_01",
             use_pixel_change=False,
             use_value_replay=False,
             use_reward_prediction=False,
             checkpoint_dir="/tmp/unreal_checkpoints",
             parallel_size=8,
             local_t_max=20,
             rmsp_alpha=0.99,
             rmsp_epsilon=0.1,
             log_file="/tmp/unreal_log/unreal_log",
             initial_alpha_low=1e-4,
             initial_alpha_high=5e-3,
             initial_alpha_log_rate=0.5,
             gamma=0.99,
             gamma_pc=0.9,
             entropy_beta=0.001,
             pixel_change_lambda=0.05,
             experience_history_size=2000,
             max_time_step=10 * 10**7,
             save_interval_step=100 * 1000,
             grad_norm_clip=40.0)
flags = namedtuple('options', flags.keys())(**flags)


def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1-rate) + log_hi * rate
    return math.exp(v)

device = "/cpu:0"
initial_learning_rate = log_uniform(flags.initial_alpha_low,
                                    flags.initial_alpha_high,
                                    flags.initial_alpha_log_rate)

action_size = Environment.get_action_size(flags.env_type,
                                        flags.env_name)

@register_trainer('unreal-tensorflow2', max_time_steps = 10 * 10**7, validation_period = None, validation_episodes = None,  episode_log_interval = 10, saving_period = 500000, save = False)
class A3CTrainer(ThreadServerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 8

    def _initialize(self, **model_kwargs):       
        self.global_network = UnrealModel(action_size,
                                          -1,
                                          flags.use_pixel_change,
                                          flags.use_value_replay,
                                          flags.use_reward_prediction,
                                          flags.pixel_change_lambda,
                                          flags.entropy_beta,
                                          device)
        self.trainers = []

        self.learning_rate_input = tf.placeholder("float")

        self.grad_applier = RMSPropApplier(learning_rate=self.learning_rate_input,
                                      decay=flags.rmsp_alpha,
                                      momentum=0.0,
                                      epsilon=flags.rmsp_epsilon,
                                      clip_norm=flags.grad_norm_clip,
                                      device=device)

        # prepare session
        config = tf.ConfigProto(log_device_placement=False,
                                allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)

        super()._initialize(**model_kwargs)

        self.sess.run(tf.global_variables_initializer())    
        

    def create_env(self, env):
        env = gym.make(**env)
        env = RewardCollector(env)
        env = ScaledFloatFrame(env)
        env = UnrealEnvBaseWrapper(env)
        return env

    def create_model(self, thread_id = -1, **model_kwargs):
        return UnrealModel(action_size,
            -1,
            flags.use_pixel_change,
            flags.use_value_replay,
            flags.use_reward_prediction,
            flags.pixel_change_lambda,
            flags.entropy_beta,
            device)

    def create_worker(self, id):
        trainer = Trainer(
            self.sess,
            id,
            self.global_network,
            initial_learning_rate,
            self.learning_rate_input,
            self.grad_applier,
            flags.env_type,
            flags.env_name,
            flags.use_pixel_change,
            flags.use_value_replay,
            flags.use_reward_prediction,
            flags.pixel_change_lambda,
            flags.entropy_beta,
            flags.local_t_max,
            flags.gamma,
            flags.gamma_pc,
            flags.experience_history_size,
            flags.max_time_step,
            device)

        return trainer