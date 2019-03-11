from abc import abstractclassmethod
import tempfile

from ..common.env import make_vec_envs, VecTransposeImage
from ..core import ThreadServerTrainer
from .trainer import A2CTrainer

class A3CTrainer(ThreadServerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = 0.99
        self.num_steps = 5
        self.entropy_coefficient = 0.01
        self.value_coefficient = 0.5
        self.max_gradient_norm = 0.5
        self.rms_alpha = 0.99
        self.rms_epsilon = 1e-5

        self.log_dir = None

    @abstractclassmethod
    def create_model(self, **model_kwargs):
        pass

    def create_env(self, env):
        self.log_dir = tempfile.TemporaryDirectory()
        envs = make_vec_envs(env, 1, 1,
                        self.gamma, self.log_dir.name, None, False)

        if len(envs.observation_space.shape) == 3:
            envs = VecTransposeImage(envs)

        return envs 

    def _finalize(self):
        self.log_dir.cleanup()
        self.log_dir = None    

    def create_worker(self, id):
        env_kwargs = dict(self._env_kwargs)
        if 'seed' in env_kwargs:
            env_kwargs['seed'] = env_kwargs['seed'] + id

        worker = A2CTrainer(self.name + '_%s' % id, env_kwargs, self._model_kwargs)
        worker.gamma = self.gamma
        worker.num_steps = self.num_steps
        worker.entropy_coefficient = self.entropy_coefficient
        worker.value_coefficient = self.value_coefficient
        worker.max_gradient_norm = self.max_gradient_norm
        worker.rms_alpha = self.rms_alpha
        worker.rms_epsilon = self.rms_epsilon

        worker.create_env = lambda **kwargs: self._sub_create_env(**env_kwargs)
        worker.create_model = self.create_model
        return worker