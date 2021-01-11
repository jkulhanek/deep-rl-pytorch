import gym
import torch
import numpy as np


class TorchWrapper(gym.vector.VectorEnvWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step_async(self, actions):
        return self.env.step_async(actions.detach().cpu().numpy())

    def step_wait(self):
        obs, rew, done, *info = self.env.step_wait()
        obs = self.from_numpy(obs)
        rew = torch.from_numpy(rew).float()
        done = torch.from_numpy(done)
        return obs, rew, done, *info

    def reset_wait(self):
        obs = self.env.reset_wait()
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


def with_collect_reward_info(env_fn):
    def _env_fn(*args, **kwargs):
        env = env_fn(*args, **kwargs)
        has_collector = False
        current_env = env
        while hasattr(current_env, 'env'):
            if current_env.__class__.__name__ == 'RewardCollector':
                has_collector = True
            if current_env == env.env:
                break
            current_env = env.env
        if not has_collector:
            env = RewardCollector(env)
        return env
    return _env_fn
