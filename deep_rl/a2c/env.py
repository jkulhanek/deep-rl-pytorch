from ..common.vec_env import DummyVecEnv, SubprocVecEnv

def create_vec_envs(thunk, num_processes):
    return SubprocVecEnv([thunk for _ in range(num_processes)]), DummyVecEnv([thunk])