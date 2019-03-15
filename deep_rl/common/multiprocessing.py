from torch.multiprocessing import Queue, Value, Process
import torch.multiprocessing as mp
from threading import Thread
from functools import partial
from abc import abstractclassmethod
from .util import serialize_function
from ..core import AbstractTrainer

def _run_process(children, report_queue, global_t, is_stopped, env_fn):
    if not hasattr(children, 'create_env') or children.create_env is None:
        children.create_env = env_fn(children)
        
    if hasattr(children, 'initialize'):
        children.initialize()

    def _process(process, *args, **kwargs):
        result = process(*args, **kwargs)
        report_queue.put(result)
        children._global_t = global_t.value
        children._is_stopped = is_stopped.value
        return result

    children.run(process = partial(_process, process = children.process))

class ProcessServerTrainer(AbstractTrainer):
    def __init__(self, name, env_kwargs, model_kwargs, **kwargs):
        super().__init__(env_kwargs = env_kwargs, model_kwargs = model_kwargs, **kwargs)
        self.name = name
        self.num_processes = 16

        self._report_queue = Queue(maxsize = 16)
        self._shared_global_t = Value('i', 0)
        self._shared_is_stopped = Value('i', False)

    @property
    def _global_t(self):
        return self._shared_global_t.value
  

    def _initialize(self, **model_kwargs):
        self.workers = [self.create_worker(i) for i in range(self.num_processes)]
        env_fn = serialize_function(self._sub_create_env, self._env_kwargs)

        # Create processes
        self.processes = [mp.Process(target = _run_process, args = (
            worker, 
            self._report_queue, 
            self._shared_global_t, 
            self._shared_is_stopped,
            env_fn)) for worker in self.workers]

        return None

    def stop(self):
        self._shared_is_stopped.value = True
        for t in self.processes:
            t.join()

    @abstractclassmethod
    def create_worker(self, id):
        pass

    def process(self, mode = 'train', **kwargs):
        assert mode == 'train'
        delta_t, epend, stats = self._report_queue.get()
        return delta_t, epend, stats

    def run(self, process, **kwargs):
        # Initialize
        self._sub_create_env = self.create_env
        self.create_env = lambda *args, **env_kwargs: None
        super().run(process, **kwargs)
        self.create_env = self._sub_create_env
        del self._sub_create_env

        # Initialize globals
        self._shared_global_t.value = 0
        self._shared_is_stopped.value = False

        # Start created threads
        for t in self.processes:
            t.start()

        while not self._shared_is_stopped.value:
            tdiff, _, _ = process(mode = 'train', context = dict())
            self._shared_global_t.value += tdiff

        for t in self.processes:
            t.join()

        self._finalize()
        return None



class ThreadServerTrainer(AbstractTrainer):
    def __init__(self, name, env_kwargs, model_kwargs, **kwargs):
        super().__init__(env_kwargs = env_kwargs, model_kwargs = model_kwargs)
        self.name = name
        self.num_processes = 16

        self._report_queue = Queue(maxsize = 16)
        self._shared_global_t = Value('i', 0)
        self._shared_is_stopped = Value('i', False)

    @property
    def _global_t(self):
        return self._shared_global_t.value
  

    def _initialize(self, **model_kwargs):
        self.workers = [self.create_worker(i) for i in range(self.num_processes)]
        env_fn = serialize_function(self._sub_create_env, self._env_kwargs)

        # Create processes
        self.processes = [Thread(target = _run_process, args = (
            worker, 
            self._report_queue, 
            self._shared_global_t, 
            self._shared_is_stopped,
            env_fn)) for worker in self.workers]

        return None

    def stop(self):
        self._shared_is_stopped.value = True
        for t in self.processes:
            t.join()

    @abstractclassmethod
    def create_worker(self, id):
        pass

    def process(self, mode = 'train', **kwargs):
        assert mode == 'train'
        delta_t, epend, stats = self._report_queue.get()
        return delta_t, epend, stats

    def run(self, process, **kwargs):
        # Initialize
        self._sub_create_env = self.create_env
        self.create_env = lambda *args, **env_kwargs: None
        super().run(process, **kwargs)
        self.create_env = self._sub_create_env
        del self._sub_create_env

        # Initialize globals
        self._shared_global_t.value = 0
        self._shared_is_stopped.value = False

        # Start created threads
        for t in self.processes:
            t.start()

        while not self._shared_is_stopped.value:
            tdiff, _, _ = process(mode = 'train', context = dict())
            self._shared_global_t.value += tdiff

        for t in self.processes:
            t.join()

        self._finalize()
        return None

