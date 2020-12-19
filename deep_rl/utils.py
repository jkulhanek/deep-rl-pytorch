import torch
import numpy as np
import dataclasses
from functools import partial
from math import ceil


@dataclasses.dataclass
class KeepTensor:
    data: torch.Tensor = None


def to_tensor(value, device):
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(value, to_tensor(value.__dict__, device))

    elif isinstance(value, list):
        return [to_tensor(x, device) for x in value]
    elif isinstance(value, tuple):
        return tuple(to_tensor(list(value), device))

    elif isinstance(value, dict):
        return {key: to_tensor(val, device) for key, val in value.items()}

    elif isinstance(value, np.ndarray):
        if value.dtype == np.bool:
            value = value.astype(np.float32)

        return torch.from_numpy(value).to(device)
    elif torch.is_tensor(value):
        return value.to(device)
    else:
        raise Exception('%s Not supported' % type(value))


def to_numpy(tensor):
    if isinstance(tensor, KeepTensor):
        return tensor.data
    elif dataclasses.is_dataclass(tensor):
        return dataclasses.replace(tensor, to_tensor(tensor.__dict__))

    elif isinstance(tensor, tuple):
        return tuple((to_numpy(x) for x in tensor))
    elif isinstance(tensor, list):
        return [to_numpy(x) for x in tensor]
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif isinstance(tensor, float) or isinstance(tensor, int):
        return tensor
    else:
        raise Exception('Not supported type %s' % type(tensor))


def pytorch_call(device):
    def wrap(function):
        def call(*args, **kwargs):
            results = function(*to_tensor(args, device), **to_tensor(kwargs, device))
            return to_numpy(results)
        return call
    return wrap


def detach_all(data):
    if isinstance(data, list):
        return [detach_all(x) for x in data]
    elif isinstance(data, tuple):
        return tuple(detach_all(list(data)))
    else:
        return data.detach()


def get_batch_size(observations):
    if isinstance(observations, (tuple, list)):
        return get_batch_size(observations[0])
    elif isinstance(observations, dict):
        return get_batch_size(next(iter(observations.values())))
    else:
        return observations.shape[0]


def batch_observations(observations):
    if isinstance(observations[0], tuple):
        return tuple(batch_observations(list(map(list, observations))))
    elif isinstance(observations[0], list):
        return list(map(lambda *x: batch_observations(x), *observations))
    elif isinstance(observations[0], dict):
        return {key: batch_observations([o[key] for o in observations]) for key in observations[0].keys()}
    else:
        return np.stack(observations, axis=1)


def print_shape(x):
    if isinstance(x, (tuple, list)):
        return '(' + ", ".join(print_shape(y) for y in x) + ')'
    return str(x.shape)


def expand_time_dimension(inputs):
    if isinstance(inputs, list):
        return [expand_time_dimension(x) for x in inputs]
    elif isinstance(inputs, tuple):
        return tuple(expand_time_dimension(list(inputs)))
    else:
        batch_size = inputs.size()[0]
        return inputs.view(batch_size, 1, *inputs.size()[1:])


def expand_time_and_batch_dimensions(inputs):
    if isinstance(inputs, list):
        return [expand_time_and_batch_dimensions(x) for x in inputs]
    elif isinstance(inputs, tuple):
        return tuple(expand_time_and_batch_dimensions(list(inputs)))
    else:
        return inputs.unsqueeze(0).unsqueeze(0)


def split_batches(num_batches, x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return np.array_split(x, num_batches)
    if isinstance(x, torch.Tensor):
        return torch.chunk(x, num_batches)
    if isinstance(x, dict):
        result = [dict()] * num_batches
        for k, x in x.items():
            for cont, xi in zip(result, split_batches(num_batches, x)):
                cont[k] = xi
        return result
    if isinstance(x, KeepTensor):
        return list(zip(KeepTensor, split_batches(num_batches, x)))
    _split_batches = partial(split_batches, num_batches)
    if isinstance(x, tuple):
        if len(x) == 0:
            return [tuple()] * num_batches
        return list(zip(*map(_split_batches, x)))
    if isinstance(x, list):
        if len(x) == 0:
            return [list()] * num_batches
        return list(map(list, zip(*map(_split_batches, x))))
    raise ValueError(f'Type {type(x)} is not supported')


def minibatch_gradient_update(inputs, compute_loss_fn, zero_grad_fn, optimize_fn, chunks=1):
    def split_inputs(inputs, chunks, axis):
        if isinstance(inputs, list):
            return list(map(list, split_inputs(tuple(inputs), chunks, axis)))
        elif isinstance(inputs, tuple):
            return list(zip(*[split_inputs(x, chunks, axis) for x in inputs]))
        else:
            return torch.chunk(inputs, chunks, axis)

    # Split inputs to chunks
    if chunks == 1:
        zero_grad_fn()
        losses = compute_loss_fn(*inputs)
        losses[0].backward()
        optimize_fn()
        return [x.item() for x in losses]

    main_inputs = split_inputs(inputs[:-1], chunks, 0)
    states_inputs = split_inputs(inputs[-1:], chunks, 1)
    if len(states_inputs) == 0:
        inputs = [x + ([],) for x in main_inputs]
    else:
        inputs = [x + y for x, y in zip(main_inputs, states_inputs)]

    # Zero gradients
    zero_grad_fn()
    total_results = None
    for minibatch in inputs:
        results = compute_loss_fn(*minibatch)
        results = list(map(lambda x: x / minibatch[1].size(0), results))
        loss = results[0]
        loss.backward()

        if total_results is None:
            total_results = results
        else:
            total_results = list(map(lambda x, y: x + y, total_results, results))

    # Optimize
    optimize_fn()
    return [x.item() for x in total_results]


class AutoBatchSizeOptimizer:
    def __init__(self, zero_grad_fn, compute_loss_fn, apply_gradients_fn):
        self._chunks = 1
        self.zero_grad_fn = zero_grad_fn
        self.compute_loss_fn = compute_loss_fn
        self.apply_gradients_fn = apply_gradients_fn

    def optimize(self, inputs):
        batch_size = inputs[1].size()[0]
        results = None
        while results is None:
            try:
                results = minibatch_gradient_update(inputs, self.compute_loss_fn, self.zero_grad_fn, self.apply_gradients_fn, self._chunks)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and self._chunks < batch_size:
                    # We will try to recover from this error
                    torch.cuda.empty_cache()
                    print('ERROR: Training failed with mini-batch size %s' % ceil(float(batch_size) / float(self._chunks)))
                    print('Trying to split the minibatch (%s -> %s)' % (self._chunks, self._chunks + 1))
                    print('Resuming training')
                    self._chunks += 1
                else:
                    raise e

        return results
