from ..common.env import RewardCollector, TransposeImage, ScaledFloatFrame
from gym.vector import AsyncVectorEnv, SyncVectorEnv
import numpy as np
import gym


class UnrealEnvBaseWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_action_reward = None
        self.observation_space = gym.spaces.Tuple((
            env.observation_space,
            gym.spaces.Box(0.0, 1.0, (env.action_space.n + 1,), dtype=np.float32)
        ))

    def reset(self):
        self.last_action_reward = np.zeros(self.action_space.n + 1, dtype=np.float32)
        return self.observation(self.env.reset())

    def step(self, action):
        observation, reward, done, stats = self.env.step(action)
        self.last_action_reward = np.zeros(self.action_space.n + 1, dtype=np.float32)
        self.last_action_reward[action] = 1.0
        self.last_action_reward[-1] = np.clip(reward, -1, 1)
        return self.observation(observation), reward, done, stats

    def observation(self, observation):
        return (observation, self.last_action_reward)


def create_unreal_env(num_processes, kwargs):
    def thunk(env):
        env = gym.make(**env)
        env = RewardCollector(env)
        env = TransposeImage(env)
        env = ScaledFloatFrame(env)
        env = UnrealEnvBaseWrapper(env)
        return env

    return AsyncVectorEnv([lambda: thunk(kwargs) for _ in range(num_processes)]), SyncVectorEnv([lambda: thunk(kwargs)])
