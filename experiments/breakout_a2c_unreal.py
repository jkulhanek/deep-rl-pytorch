import environments
import numpy as np
import gym

from deep_rl import register_trainer
from deep_rl.a2c_unreal import UnrealTrainer
from deep_rl.a2c_unreal.model import UnrealModel
from deep_rl.common.schedules import LinearSchedule
from deep_rl.common.vec_env import DummyVecEnv, SubprocVecEnv
from baselines.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, ClipRewardEnv, WarpFrame, ScaledFloatFrame
from deep_rl.common.env import VecFrameStack, VecTransposeImage, RewardCollector, TransposeImage
from deep_rl.a2c_unreal.util import UnrealEnvBaseWrapper

@register_trainer(max_time_steps = 40e6, validation_period = 50000, validation_episodes = 20,  episode_log_interval = 10, saving_period = 500000, save = True)
class Trainer(UnrealTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 16
        self.max_gradient_norm = 0.5
        self.rms_alpha = 0.99
        self.rms_epsilon = 1e-5
        self.data_parallel = True
        self.learning_rate = 7e-4

        self.num_steps = 20
        self.gamma = .99
        self.allow_gpu = True
        

        self.rp_weight = 1.0
        self.pc_weight = 0.05
        self.vr_weight = 1.0
        
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

def default_args():
    return dict(
        env_kwargs = dict(id = 'BreakoutNoFrameskip-v4'),
        model_kwargs = dict()
    )