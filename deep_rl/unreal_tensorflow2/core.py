from abc import abstractclassmethod
from deep_rl.core import AbstractTrainer
from deep_rl.common.train_wrappers import MetricContext
from multiprocessing import Queue, Value

from threading import Thread
from functools import partial

class ThreadServerTrainer(AbstractTrainer):
    def __init__(self, name, env_kwargs, model_kwargs, **kwargs):
        super().__init__(env_kwargs = env_kwargs, model_kwargs = model_kwargs, **kwargs)
        self.name = name
        self._report_queue = Queue(maxsize = 16)
        self._shared_global_t = Value('i', 0)
        self._shared_is_stopped = Value('i', False)
        self._num_workers = 16

    @property
    def _global_t(self):
        return self._shared_global_t.value
    
    def _child_run(self, id, env_kwargs, model_kwargs):
        worker = self.create_worker(id, env_kwargs, model_kwargs)
        def _process(process, *args, **kwargs):
            result = process(*args, **kwargs)
            self._report_queue.put(result)
            worker._shared_global_t = self._shared_global_t.value
            worker._shared_is_stopped = self._shared_is_stopped.value
            return result

        worker.run(process = partial(_process, process = worker.process))      

    def _initialize(self, **model_kwargs):
        self.workers = [Thread(target=self._child_run,args=(i, self._env_kwargs, self._model_kwargs)) for i in range(self._num_workers)]
        for t in self.workers:
            t.start()

    def stop(self):
        self._shared_is_stopped.value = True
        for t in self.workers:
            t.join()

    def create_env(self, *args, **kwargs):
        return None

    @abstractclassmethod
    def create_worker(self, id, env_kwargs, model_kwargs):
        pass


    def process(self, mode = 'train', **kwargs):
        assert mode == 'train'
        delta_t, epend, stats = self._report_queue.get()
        return delta_t, epend, stats

    def run(self, process, **kwargs):
        super().run(process, **kwargs)
        self._shared_global_t.value = 0
        self._shared_is_stopped.value = False
        while not self._shared_is_stopped.value:
            tdiff, _, _ = process(mode = 'train', context = dict())
            self._shared_global_t.value += tdiff

        self._finalize()
        return None
        



