from abc import abstractclassmethod
import tempfile

from ..common.multiprocessing import ProcessServerTrainer
from .trainer import A3CWorker
from ..common.util import serialize_function
from ..optim.shared_rmsprop import SharedRMSprop

class A3CTrainer(ProcessServerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.gamma = 0.99
        self.num_steps = 5
        self.entropy_coefficient = 0.01
        self.value_coefficient = 0.5
        self.max_gradient_norm = 0.5
        self.learning_rate = 7e-4
        self.rms_alpha = 0.99
        self.rms_epsilon = 1e-5
        self.log_dir = None

    @abstractclassmethod
    def create_model(self, **model_kwargs):
        pass

    def _initialize(self, **model_kwargs):
        if hasattr(self, 'schedules') and 'learning_rate' in self.schedules:
            raise Exception('Learning rate schedule is not yet implemented for a3c')

        self.model = self.create_model(**model_kwargs).share_memory()
        self.optimizer = SharedRMSprop(self.model.parameters(), self.learning_rate, self.rms_alpha, self.rms_epsilon).share_memory()
        super()._initialize(**model_kwargs)
        return self.model   

    def create_worker(self, id):
        model_fn = serialize_function(self.create_model, **self._model_kwargs)
        worker = A3CWorker(self.model, self.optimizer, model_fn)
        worker.gamma = self.gamma
        worker.num_steps = self.num_steps
        worker.entropy_coefficient = self.entropy_coefficient
        worker.value_coefficient = self.value_coefficient
        worker.max_gradient_norm = self.max_gradient_norm
        return worker