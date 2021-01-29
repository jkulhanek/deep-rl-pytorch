import numpy as np
from deep_rl.utils.tensor import stack_observations, unstack, cat


def merge_batches(*batches, **kwargs):
    axis = kwargs.get('axis', 0)
    batches = [x for x in batches if x is not None and (not isinstance(x, list) or len(x) != 0)]
    if len(batches) == 1:
        return batches[0]
    elif len(batches) == 0:
        return None

    return cat(batches, axis)


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

    def sample(self, getter, index, size):
        batch = []
        for i in range(index - self.sequence_length + 1, index + 1):
            batch.append(getter(i))

        batch = stack_observations(batch, 0)
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

    def sample(self, getter, index, size):
        if index == size - 1:
            # Cannot allow the last index to be selected
            raise NewSelectionException()

        batch = []
        for i in range(index - self.sequence_length + 1, index + 1):
            batch.append(getter(i))

        batch.append(getter(index + 1))
        batch = stack_observations(batch, 0)
        return batch


class SequenceStorage:
    def __len__(self):
        return len(self.storage)

    def __init__(self, size, samplers=[]):
        self.samplers = samplers

        self.size = size
        self.storage = []

        self.selector_data = np.zeros((self.size, len(self.samplers),), dtype=np.bool)
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

    def count(self, sampler):
        return self.selector_lengths[sampler]

    def sample(self, sampler):
        result = None
        trials = 0
        while trials < 200 and result is None:
            try:
                sampler_obj = self.samplers[sampler]
                index = np.random.choice(np.where(self.selector_data[:, sampler])[0])
                result = sampler_obj.sample(lambda i: self[i], (index - self.tail) % self.size, len(self.storage))
            except NewSelectionException:
                pass

            trials += 1

        return result

    @property
    def full(self):
        return len(self.storage) == self.size


class BatchSequenceStorage:
    def __init__(self, num_storages, single_size, samplers=[]):
        self.samplers = samplers
        self.storages = [SequenceStorage(single_size, samplers=samplers) for _ in range(num_storages)]

    def insert(self, observations, actions, rewards, terminals):
        batch = (observations, actions, rewards, terminals)
        rows = unstack(batch, axis=0)
        for storage, row in zip(self.storages, rows):
            storage.insert(*row)

    @property
    def full(self):
        return all([x.full for x in self.storages])

    def counts(self, sequencer):
        return [x.count(sequencer) for x in self.storages]

    def sample(self, sampler, batch_size=None):
        if batch_size is None:
            batch_size = len(self.storages)

        if batch_size == 0:
            return []

        probs = np.array([x.selector_lengths[sampler] for x in self.storages], dtype=np.float32)
        if np.sum(probs) == 0:
            print("WARNING: Could not sample data from storage")
            return None

        probs = probs / np.sum(probs)
        selected_sources = np.random.choice(np.arange(len(self.storages)), size=(batch_size,), p=probs)

        sequences = []
        for source in selected_sources:
            sd = self.storages[source].sample(sampler)
            if sd is None:
                return None

            sequences.append(sd)

        return stack_observations(sequences, 0)


def default_samplers(sequence_length):
    return [
        PlusOneSampler(sequence_length),
        LambdaSampler(4, lambda _, get: get(-1)[2].item() == 0.0),
        LambdaSampler(4, lambda _, get: get(-1)[2].item() != 0.0)
    ]


class ExperienceReplay(SequenceStorage):
    def __init__(self, size, sequence_length):
        super().__init__(size, samplers=default_samplers(sequence_length))

    def sample_sequence(self):
        return self.sample(0)

    def _choose_rp_sequencer(self):
        if self.count(1) < self.samplers[1].sequence_length:
            return 2

        if self.count(2) < self.samplers[2].sequence_length:
            return 1

        if np.random.randint(2) == 0:
            return 1  # from zero 1/3 probability
        else:
            return 2

    def sample_rp_sequence(self):
        return self.sample(self._choose_rp_sequencer())


class BatchExperienceReplay(BatchSequenceStorage):
    def __init__(self, num_processes, size, sequence_length):
        super().__init__(num_processes, size, samplers=default_samplers(sequence_length))

    def sample_sequence(self):
        return self.sample(0)

    def _num_rp_zeros(self):
        fromzeros = np.random.binomial(len(self.storages), 0.3333)  # Probability of selecting zero sequence
        zeroenvs = sum(self.counts(1))
        nonzeroenvs = sum(self.counts(2))
        fromzeros = min(fromzeros, zeroenvs)
        fromzeros = max(fromzeros, len(self.storages) - nonzeroenvs)
        return fromzeros

    def sample_rp_sequence(self):
        fromzeros = self._num_rp_zeros()
        sampler1_batch = self.sample(1, batch_size=fromzeros)
        sampler2_batch = self.sample(2, len(self.storages) - fromzeros)
        if sampler1_batch is None or sampler2_batch is None:
            return None

        return merge_batches(sampler1_batch, sampler2_batch)
