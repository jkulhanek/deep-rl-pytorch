import torch
from collections import OrderedDict


def to_tensor(value, dtype=torch.float32):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().to(dtype)
    return torch.tensor(value, dtype=dtype)


class Metric:
    def __init__(self, is_distributed=False):
        self.is_distributed = is_distributed
        self.reset_states()

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            return self.update_state(*args, **kwargs)
        else:
            return self.report()


class Mean(Metric):
    def reset_states(self):
        self.cumsum = 0.0
        self.samples = 0.0

    def update_state(self, value, weight=None):
        value = to_tensor(value, dtype=torch.float32)
        weight = to_tensor(weight, dtype=torch.float32)
        if weight is None and hasattr(value, 'shape'):
            weight = torch.prod(to_tensor(value.shape))
            value = value.sum()
        if weight is None:
            weight = to_tensor(1.0, dtype=torch.float32)

        self.samples += weight.cpu().item()
        self.cumsum += value.cpu().item()

    def report(self):
        if self.samples == 0:
            return 0
        return self.cumsum / self.samples


class AccumulatedMetric(Metric):
    def __init__(self, accumulate_fn=None, **kwargs):
        if not hasattr(self, '_accumulate'):
            self._accumulate = accumulate_fn
        super().__init__(**kwargs)

    def reset_states(self):
        self.value = 0.0

    def update_state(self, value):
        self.value = self._accumulate(self.value, to_tensor(value).cpu().item())

    def report(self):
        return self.value


class LastValue(AccumulatedMetric):
    def _accumulate(self, acc, value):
        return value


class MetricsContext:
    def __init__(self, **kwargs):
        self.metrics = OrderedDict(**kwargs)

    def log(self, name, *args, **kwargs):
        if name not in self.metrics:
            self.metrics[name] = Mean()
        self.metrics[name](*args, **kwargs)

    def collect(self, is_distributed=False):
        vals = {k: val() for k, val in self.metrics.items()}
        distributed_vals = []
        distributed_keys = []
        for name, m in self.metrics.items():
            if m.is_distributed:
                distributed_vals.append(vals[name])
                distributed_keys.append(name)
            m.reset_states()
        if is_distributed:
            values = torch.tensor(distributed_vals, dtype=torch.float32).cuda()
            torch.distributed.all_reduce(values)
            values /= torch.distributed.get_world_size()
            values = list(values.cpu())
            for k, val in zip(distributed_keys, values):
                vals[k] = val
        return vals
