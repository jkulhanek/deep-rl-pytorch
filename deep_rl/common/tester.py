import gym
import numpy as np
from .vec_env import VecEnv, flatten_observations
import tempfile

class TestingEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        gym.spaces.Box().sample()
        return self.observation_space.sample(), 0.0, False, dict()

class TestFinished(Exception):
    def __init__(self):
        super().__init__()

class TestingVecEnv(VecEnv):
    def __init__(self, n_processes, observation_space, action_space):
        super().__init__(n_processes, observation_space, action_space)

    def _sample(self):
        return flatten_observations([self.observation_space.sample() for _ in range(self.num_envs)])

    def reset(self):
        return self._sample()

    def step_async(self, action):
        pass

    def step_wait(self):
        return self._sample(), np.zeros((self.num_envs,), dtype = np.float32), np.zeros((self.num_envs,), dtype = np.bool), [dict() for _ in range(self.num_envs)]

def test_trainer(trainer):
    create_env = trainer.unwrapped.create_env
    def wrap_env(env):
        if hasattr(env, 'step_wait'):
            # Vector environment
            parent = env
            while hasattr(env, 'venv'):
                parent = env
                env = env.venv

            parent.venv = TestingVecEnv(env.num_envs, env.observation_space, env.action_space)

        else:
            parent = env
            while hasattr(env, 'env'):
                parent = env
                env = env.env

            parent.env = TestingEnv(env.observation_space, env.action_space)
        return env


    def _create_env(*args, **kwargs):
        env = create_env(*args, **kwargs)
        if hasattr(trainer.unwrapped, 'validation_env'):
            trainer.unwrapped.validation_env = wrap_env(trainer.unwrapped.validation_env)

        return wrap_env(env)

    assert trainer.__class__.__name__ == 'CompiledTrainer'
    with tempfile.TemporaryDirectory() as tmpdir:

        # Replace create env with dummy env
        trainer.unwrapped._create_env = _create_env
        
        t = trainer
        while hasattr(t, 'trainer'):
            # Redirect saving to temp directory
            if t.__class__.__name__ == 'SaveWrapper':
                t.model_root_directory = tmpdir.name

            t = t.trainer

        process_base = trainer.process
        def process(*args, **kwargs):
            res = process_base(*args, **kwargs)
            
            # Run 30 iterations of process
            if trainer.unwrapped._global_t > 30:
                raise TestFinished()
            return res
        trainer.process = process
        
        try:
            trainer.run()
            trainer.run = lambda *args, **kwargs: print('ERROR: Cannot run tested trainer')
        except TestFinished:
            pass

