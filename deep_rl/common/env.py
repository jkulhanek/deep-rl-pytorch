import os
import time
import gym
import numpy as np
import torch
from gym.spaces.box import Box
from copy import copy

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from .vec_env import VecEnvWrapper, SubprocVecEnv, DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_
from baselines.common.vec_env.vec_frame_stack import VecFrameStack


try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

try:
    import environments
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, add_timestep, allow_early_resets):
    env_id = env_id
    if isinstance(env_id, dict):
        allow_early_resets = env_id.get('allow_early_resets', allow_early_resets)
        env_id = env_id.get('id')        
        
    def _thunk():
        if callable(env_id):
            env = env_id()
        elif env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)

        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if add_timestep and len(
                obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimestep(env)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
                                allow_early_resets=allow_early_resets)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env)
        
        return env

    return _thunk

def make_vec_envs(env_name, seed, num_processes, gamma, log_dir, add_timestep, allow_early_resets, num_frame_stack = None):
    if isinstance(env_name, dict):
        if num_frame_stack is None and 'num_frame_stack' in env_name:
            num_frame_stack = env_name.get('num_frame_stack')

    envs = [make_env(env_name, seed, i, log_dir, add_timestep, allow_early_resets)
            for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)


    if num_frame_stack is not None:
        if num_frame_stack != 1:
            envs = VecFrameStack(envs, num_frame_stack)
    elif len(envs.observation_space.shape) == 3:
        envs = VecFrameStack(envs, 4)

    return envs


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:0] = 0
        return observation


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)



class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, %s, must be dim3" % str(op)
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = self.transpose_space(self.observation_space)

    def transpose_space(self, space):
        if space.__class__.__name__ == 'Box':
            if len(space.shape) == 3:
                obs_shape = space.shape
                return Box(
                    space.low[0, 0, 0],
                    space.high[0, 0, 0],
                    [
                        obs_shape[self.op[0]],
                        obs_shape[self.op[1]],
                        obs_shape[self.op[2]]],
                    dtype=space.dtype)
            return space
        elif space.__class__.__name__ == 'Tuple':
            return gym.spaces.Tuple(tuple(map(self.transpose_space, space.spaces)))
        else:
            raise Exception('Environment type is not supported')

    def transpose_observation(self, ob, space):
        if space.__class__.__name__ == 'Box':
            if len(space.shape) == 3:
                return ob if ob is None else np.transpose(ob, axes = self.op)
            return ob
        elif space.__class__.__name__ == 'Tuple':
            return tuple(map(self.transpose_observation, zip(ob, space.spaces)))
        else:
            raise Exception('Environment type is not supported')

    def observation(self, ob):
        return self.transpose_observation(ob, self.env.observation_space)

class VecTransposeImage(VecEnvWrapper):
    def __init__(self, venv, transpose = [2, 0, 1]):
        if venv.observation_space.__class__.__name__ != 'Box':
            raise Exception('Env type %s is not supported' % venv.__class__.__name__)

        self._transpose = (0,) + tuple([1 + x for x in transpose])
        obs_space = copy(venv.observation_space)
        obs_space.shape = tuple([obs_space.shape[i] for i in transpose])

        super().__init__(venv, observation_space=obs_space)

    def reset(self):
        obs = self.venv.reset()
        obs = np.transpose(obs, self._transpose)
        return obs

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = np.transpose(obs, self._transpose)
        return obs, reward, done, info


class VecNormalize(VecNormalize_):

    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = self.transform_space(env.observation_space)

    def transform_space(self, space):
        if space.__class__.__name__ == 'Box':
            if len(space.shape) == 3 and space.dtype == np.uint8:
                return gym.spaces.Box(low=0.0, high=1.0, shape = space.shape, dtype = np.float32)
            return space
        elif space.__class__.__name__ == 'Tuple':
            return gym.spaces.Tuple(tuple(map(self.transform_space, space.spaces)))
        else:
            raise Exception('Environment type is not supported')

    def transform_observation(self, ob, space):
        if space.__class__.__name__ == 'Box':
            if len(space.shape) == 3 and space.dtype == np.uint8:
                return ob if ob is None else np.array(ob).astype(np.float32) / 255.0
            return ob
        elif space.__class__.__name__ == 'Tuple':
            return tuple(map(self.transform_observation, zip(ob, space.spaces)))
        else:
            raise Exception('Environment type is not supported')

    def observation(self, ob):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        if ob is None:
            return None


        return self.transform_observation(ob, self.env.observation_space)


class RewardCollector(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self.rewards = None

    def reset(self, **kwargs):
        self.reset_state()
        return self.env.reset(**kwargs)

    def reset_state(self):
        self.rewards = []

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        self.update(ob, rew, done, info)
        return (ob, rew, done, info)

    def update(self, ob, rew, done, info):
        assert isinstance(info, dict)

        self.rewards.append(rew)       
        info['reward'] = rew
        if done:
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen}
            info['episode'] = epinfo
