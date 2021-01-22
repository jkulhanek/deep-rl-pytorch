import torch
from functools import partial
import dataclasses


def stack_observations(observations, axis=1):
    if isinstance(observations[0], torch.Tensor):
        return torch.stack(observations, axis=axis)
    elif isinstance(observations[0], tuple):
        return tuple(map(lambda *x: stack_observations(x, axis=axis), *observations))
    elif isinstance(observations[0], list):
        return list(map(lambda *x: stack_observations(x, axis=axis), *observations))
    elif isinstance(observations[0], dict):
        return {key: stack_observations([o[key] for o in observations], axis=axis) for key in observations[0].keys()}
    elif isinstance(observations[0], dict):
        return {k: stack_observations([x[k] for x in observations], axis=axis) for k in observations[0]}
    elif dataclasses.is_dataclass(observations):
        return dataclasses.replace(observations, **stack_observations(observations.__dict__, axis=axis))
    else:
        raise ValueError(f'Value of type {type(observations[0])} is not supported')


def get_batch_size(x, axis=0):
    if isinstance(x, torch.Tensor):
        return x.shape[axis]
    if isinstance(x, (tuple, list)):
        return get_batch_size(x[0], axis)
    if isinstance(x, dict):
        return get_batch_size(next(iter(x.values())), axis)
    if dataclasses.is_dataclass(x):
        return get_batch_size(x.__dict__, axis)
    raise ValueError(f'Value of type {type(x)} is not supported')


def unstack(x, axis=0, _size=None):
    if _size is None:
        _size = get_batch_size(x, axis=axis)

    def _split_chunks(x):
        if x is None:
            return [None] * _size
        elif isinstance(x, torch.Tensor):
            return torch.unbind(x, axis)
        elif isinstance(x, tuple):
            if len(x) == 0:
                return [tuple()] * _size
            return list(zip(*map(_split_chunks, x)))
        elif isinstance(x, list):
            if len(x) == 0:
                return [list()] * _size
            return list(map(list, zip(*map(_split_chunks, x))))
        elif isinstance(x, dict):
            data = {k: _split_chunks(v) for k, v in x.items()}
            output = []
            if len(data) == 0:
                return [dict()] * _size
            else:
                for i in range(_size):
                    output.append({k: v[i] for k, v in data.items()})
            return output
        elif dataclasses.is_dataclass(x):
            return dataclasses.replace(x, **_split_chunks(x.__dict__))
        else:
            raise ValueError(f'Type {type(x)} is not supported')
    return _split_chunks(x)


def cat(batches, axis):
    if isinstance(batches[0], torch.Tensor):
        return torch.cat(batches, axis)

    elif isinstance(batches[0], tuple):
        return tuple([cat([x[i] for x in batches], axis=axis) for i in range(len(batches[0]))])

    elif isinstance(batches[0], list):
        return [cat([x[i] for x in batches], axis=axis) for i in range(len(batches[0]))]
    elif isinstance(batches[0], dict):
        return {k: cat([x[k] for x in batches], axis=axis) for k in batches[0].keys()}
    elif dataclasses.is_dataclass(batches[0]):
        return dataclasses.replace(batches[0], **cat([x.__dict__ for x in batches], axis=axis))
    else:
        raise Exception('Type not supported')


def tensor_map(fn, value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, list):
        return list(map(partial(tensor_map, fn), value))
    if isinstance(value, tuple):
        return tuple(map(partial(tensor_map, fn), value))
    if isinstance(value, dict):
        _map = partial(tensor_map, fn)
        return {k: _map(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(value, **tensor_map(fn, value.__dict__))
    raise ValueError(f'Value of type {type(value)} is not supported')


def to_device(value, device):
    return tensor_map(lambda x: x.to(device), value)


detach_all = partial(tensor_map, lambda x: x.detach())
expand_time_dimension = partial(tensor_map, lambda x: torch.unsqueeze(x, 1))
