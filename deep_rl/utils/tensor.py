import torch
from functools import partial
import dataclasses


def to_device(value, device):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return list(map(partial(to_device, device=device), value))
    if isinstance(value, tuple):
        return tuple(map(partial(to_device, device=device), value))
    if isinstance(value, dict):
        _to_device = partial(to_device, device=device)
        return {k: _to_device(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(value, **to_device(value.__dict__, device))
    raise ValueError(f'Value of type {type(value)} is not supported')


def detach_all(data):
    if isinstance(data, list):
        return [detach_all(x) for x in data]
    elif isinstance(data, tuple):
        return tuple(detach_all(list(data)))
    else:
        return data.detach()


def stack_observations(observations):
    if isinstance(observations[0], tuple):
        return tuple(stack_observations(list(map(list, observations))))
    elif isinstance(observations[0], list):
        return list(map(lambda *x: stack_observations(x), *observations))
    elif isinstance(observations[0], dict):
        return {key: stack_observations([o[key] for o in observations]) for key in observations[0].keys()}
    else:
        return torch.stack(observations, axis=1)


def expand_time_dimension(inputs):
    if isinstance(inputs, list):
        return [expand_time_dimension(x) for x in inputs]
    elif isinstance(inputs, tuple):
        return tuple(expand_time_dimension(list(inputs)))
    else:
        batch_size = inputs.size()[0]
        return inputs.view(batch_size, 1, *inputs.size()[1:])
