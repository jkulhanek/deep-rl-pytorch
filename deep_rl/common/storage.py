import numpy as np


def batch_items(items):
    if isinstance(items[0], tuple):
        return tuple(map(batch_items, zip(*items)))

    elif isinstance(items[0], list):
        return list(map(batch_items, zip(*items)))

    else:
        return np.stack(items)
        
class NewSelectionException(Exception):
    pass

class SequenceSampler:
    def __init__(self, sequence_length):
        self._sequence_length = sequence_length
    
    def is_allowed(self, sequence_length, getter):
        return sequence_length >= self.sequence_length

    @property
    def sequence_length(self):
        return self._sequence_length

    def sample(self, getter, index):
        batch = []
        for i in range(index - self.sequence_length + 1, index + 1):
            batch.append(getter(i))

        batch = batch_items(batch)
        return batch

class LambdaSampler(SequenceSampler):
    def __init__(self, sequence_length, function):
        super().__init__(sequence_length)
        self.selector = function

    def is_allowed(self, sequence_length, getter):
        return super().is_allowed(sequence_length, getter) and self.selector(sequence_length, getter)

class PlusOneSampler(SequenceSampler):
    def __init__(self, sequence_length):
        super().__init__(sequence_length)

    def sample(self, getter, index):
        if getter(-1) == getter(index):
            # Cannot allow the last index to be selected
            raise NewSelectionException()

        batch = []
        for i in range(index - self.sequence_length + 1, index + 1):
            batch.append(getter(i))
        
        batch.append(getter(index + 1))

        batch = batch_items(batch)
        return batch


class SequenceStorage:
    def __len__(self):
        return len(self.storage)

    def __init__(self, size, sequence_length, samplers = []):
        self.samplers = samplers

        self.size = size
        self.storage = []

        self.selector_data = np.zeros((self.size, len(self.samplers),), dtype = np.bool)
        self.selector_lengths = [0 for _ in range(len(self.samplers))]

        self.tail = 0
        self.episode_length = 0

    def remove_samplers(self, index):
        for i, sampler in enumerate(self.samplers):
            positive = self.selector_data[index, i]
            self.selector_data[index, i] = False
            if positive:
                self.selector_lengths[i] -= 1

            # Remove items with shorted sequences
            if self.selector_data[(index + sampler.sequence_length - 1) % self.size, i]:
                self.selector_data[(index + sampler.sequence_length - 1) % self.size, i] = False
                self.selector_lengths[i] -= 1

    def insert_samplers(self, index, ctx):
        for i, sampler in enumerate(self.samplers):
            positive = sampler.is_allowed(*ctx)
            self.selector_data[index, i] = positive

            if positive:
                self.selector_lengths[i] += 1


    def __getitem__(self, index):
        position = (self.tail + index) % self.size if len(self) == self.size else index
        return self.storage[position]

    def insert(self, observation, action, reward, terminal):
        row = (observation, action, reward, terminal)
        self.episode_length += 1
        ctx = (self.episode_length, lambda i: self[i])

        if len(self) == self.size:
            # Update samplers
            self.remove_samplers(self.tail)

            self.storage[self.tail] = row
            old_tail = self.tail
            self.tail = (self.tail + 1) % self.size

            self.insert_samplers(old_tail, ctx)
        else:
            self.storage.append(row)
            self.insert_samplers(len(self) - 1, ctx)

        if terminal:
            self.episode_length = 0


    def sample(self, sampler):
        result = None
        trials = 10
        while trials > 0 and result is None:
            try:
                sampler_obj = self.samplers[sampler]
                index = np.random.choice(np.where(self.selector_data[:, sampler])[0])
                result = sampler_obj.sample(lambda i: self[i], (index - self.tail) % self.size)
                trials -= 1
            except NewSelectionException:
                pass

        return result
            