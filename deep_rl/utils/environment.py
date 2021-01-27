import gym
import gym.spaces
import torch
import numpy as np
from functools import partial


class TransposeWrapper(gym.ObservationWrapper):
    def __init__(self, env, transpose=[2, 0, 1]):
        super().__init__(env)
        self.op = transpose
        self.observation_space = self.transpose_space(self.observation_space)

    def transpose_space(self, space):
        if space.__class__.__name__ == 'Box':
            if len(space.shape) == 3:
                obs_shape = space.shape
                return gym.spaces.Box(
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
                return ob if ob is None else np.transpose(ob, axes=self.op)
            return ob
        elif space.__class__.__name__ == 'Tuple':
            return tuple(map(lambda x: self.transpose_observation(*x), zip(ob, space.spaces)))
        else:
            raise Exception('Environment type is not supported')

    def observation(self, ob):
        return self.transpose_observation(ob, self.env.observation_space)


class TorchWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, actions):
        obs, *info = self.env.step(actions.detach().cpu().numpy())
        obs = self.from_numpy(obs)
        return obs, *info

    def reset(self):
        obs = self.env.reset()
        obs = self.from_numpy(obs)
        return obs

    @staticmethod
    def from_numpy(obj):
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        if isinstance(obj, list):
            return list(map(TorchWrapper.from_numpy, obj))
        if isinstance(obj, tuple):
            return list(map(TorchWrapper.from_numpy, obj))
        if isinstance(obj, dict):
            return {k: TorchWrapper.from_numpy(v) for k, v in obj.items()}


class VectorTorchWrapper(gym.vector.VectorEnvWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step_async(self, actions):
        return self.env.step_async(actions.detach().cpu().numpy())

    def step_wait(self):
        obs, rew, done, *info = self.env.step_wait()
        obs = TorchWrapper.from_numpy(obs)
        rew = torch.from_numpy(rew).float()
        done = torch.from_numpy(done)
        return obs, rew, done, *info

    def reset_wait(self):
        obs = self.env.reset_wait()
        obs = TorchWrapper.from_numpy(obs)
        return obs


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


def with_wrapper(env_fn, Wrapper):
    def _env_fn(*args, **kwargs):
        env = env_fn(*args, **kwargs)
        has_collector = False
        current_env = env
        while hasattr(current_env, 'env'):
            if isinstance(current_env, Wrapper):
                has_collector = True
            if current_env == env.env:
                break
            current_env = env.env
        if not has_collector:
            env = Wrapper(env)
        return env
    return _env_fn


with_collect_reward_info = partial(with_wrapper, Wrapper=RewardCollector)
