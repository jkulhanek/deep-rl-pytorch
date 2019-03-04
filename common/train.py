import abc
import gym
import threading
import os

class AbstractTrainer:
    def __init__(self, env_kwargs, model_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.env = None
        self._env_kwargs = env_kwargs
        self.model = None
        self._model_kwargs = model_kwargs
        self.name = 'trainer'

        self.is_initialized = False
        pass

    def save(self, path):
        model = self.model           
        model.save_weights(path + '/%s-weights.h5' % self.name)
        with open(path + '/%s-model.json' % self.name, 'w+') as f:
            f.write(model.to_json())
            f.flush()

    def create_env(self, env):
        if isinstance(env, dict):
            env = gym.make(**env)

        return env

    @abc.abstractclassmethod
    def _initialize(self, **model_kwargs):
        pass

    def _finalize(self):
        pass

    @abc.abstractclassmethod
    def process(self, **kwargs):
        pass

    def run(self, process, **kwargs):
        if process is None:
            raise Exception('Must be compiled before run')

        self.env = self.create_env(self._env_kwargs)
        self.model = self._initialize(**self._model_kwargs) 
        return None

    def __repr__(self):
        return '<%sTrainer>' % self.name

    def compile(self, compiled_agent = None, **kwargs):
        if compiled_agent is None:
            compiled_agent = CompiledTrainer(self)

        return compiled_agent


class AbstractTrainerWrapper(AbstractTrainer):
    def __init__(self, trainer, *args, **kwargs):
        self.trainer = trainer
        self.unwrapped = trainer.unwrapped if hasattr(trainer, 'unwrapped') else trainer
        self.summary_writer = trainer.summary_writer if hasattr(trainer, 'summary_writer') else None

    def process(self, **kwargs):
        return self.trainer.process(**kwargs)

    def stop(self, **kwargs):
        self.trainer.stop(**kwargs)

    def run(self, process, **kwargs):
        return self.trainer.run(process, **kwargs)

    def save(self, path):
        self.trainer.save(path)

class CompiledTrainer(AbstractTrainerWrapper):
    def __init__(self, target, *args, **kwargs):
        super().__init__(target, *args, **kwargs)
        self.process = target.process

    def run(self, **kwargs):
        return self.trainer.run(self.process)

    def __repr__(self):
        return '<Compiled %s>' % self.trainer.__repr__()


class SingleTrainer(AbstractTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._global_t = None
        pass

    def run(self, process, **kwargs):
        super().run(process, **kwargs)
        self._global_t = 0
        self._is_stopped = False
        while not self._is_stopped:
            tdiff, _, _ = process(mode = 'train', context = dict())
            self._global_t += tdiff

        self._finalize()
        return None