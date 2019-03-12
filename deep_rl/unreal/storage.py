from queue import deque
import numpy as np
from collections import namedtuple
from ..common.storage import SequenceStorage, BatchSequenceStorage, PlusOneSampler, LambdaSampler, batch_items, merge_batches

def default_samplers(sequence_length):
    return [
        PlusOneSampler(sequence_length),
        LambdaSampler(4, lambda _, get: get(-1)[2] == 0.0),
        LambdaSampler(4, lambda _, get: get(-1)[2] != 0.0)
    ]

class ExperienceReplay(SequenceStorage):
    def __init__(self, size, sequence_length):
        super().__init__(size, samplers = default_samplers(sequence_length))

    def sample_sequence(self):
        return self.sample(0)

    def sample_rp_sequence(self):
        if np.random.randint(2) == 0:
            from_zero = True
        else:
            from_zero = False

        return self.sample(1 if from_zero else 2)

class BatchExperienceReplay(BatchSequenceStorage):
    def __init__(self, num_processes, size, sequence_length):
        super().__init__(num_processes, size, samplers = default_samplers(sequence_length))

    def sample_sequence(self):
        return self.sample(0)

    def sample_rp_sequence(self):
        fromzeros = np.random.binomial(len(self.storages), 0.3333) # Probability of selecting zero sequence        
        sampler1_batch = self.sample(1, batch_size = fromzeros)
        sampler2_batch = self.sample(2, len(self.storages) - fromzeros)
        return merge_batches(sampler1_batch, sampler2_batch)
        
