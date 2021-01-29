import torch
from torch import nn
from deep_rl.utils.tensor import tensor_map
from functools import partial


def forward_masked_rnn(inputs, masks, states, forward_rnn):
    def mask_states(states, mask):
        if isinstance(states, tuple):
            return tuple(mask_states(list(states), mask))
        elif isinstance(states, list):
            return [mask_states(x, mask) for x in states]
        else:
            return states * mask.view(1, -1, 1)

    has_zeros = ((masks[:, 1:] == 0.0)
                 .any(dim=0)
                 .nonzero(as_tuple=False)
                 .squeeze()
                 .cpu())

    T = masks.size()[1]

    # +1 to correct the masks[1:]
    if has_zeros.dim() == 0:
        # Deal with scalar
        has_zeros = [has_zeros.item() + 1]
    else:
        has_zeros = (has_zeros + 1).numpy().tolist()

    # add t=0 and t=T to the list
    has_zeros = [0] + has_zeros + [T]
    outputs = []

    for i in range(len(has_zeros) - 1):
        # We can now process steps that don't have any zeros in masks together!
        # This is much faster
        start_idx = has_zeros[i]
        end_idx = has_zeros[i + 1]

        rnn_scores, states = forward_rnn(
            inputs[:, start_idx:end_idx],
            mask_states(states, masks[:, start_idx])
        )

        outputs.append(rnn_scores)

    # assert len(outputs) == T
    # x is a (N, T, -1) tensor
    outputs = torch.cat(outputs, dim=1)

    # flatten
    return outputs, states


switch_time_batch = partial(tensor_map, lambda tensor: tensor.permute(1, 0, *range(2, len(tensor.size()))))


def rnn_call_switch_axes(forward):
    def call(*args):
        args = list(args)
        args[-1] = switch_time_batch(args[-1])
        results = forward(*args)
        results[-1] = switch_time_batch(results[-1])
        return results
    return call


def forward_masked_rnn_transposed(inputs, masks, states, forward_rnn):
    states = switch_time_batch(states)
    outputs, states = forward_masked_rnn(inputs, masks, states, forward_rnn)
    states = switch_time_batch(states)
    return outputs, states


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class TimeDistributed(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, *args):
        batch_shape = args[0].size()[:2]
        args = [x.view(-1, *x.size()[2:]) for x in args]
        results = self.inner(*args)

        def reshape_res(x):
            return x.view(*(batch_shape + x.size()[1:]))

        if isinstance(results, list):
            return [reshape_res(x) for x in results]
        elif isinstance(results, tuple):
            return tuple([reshape_res(x) for x in results])
        else:
            return reshape_res(results)


class MaskedRNN(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, inputs, masks, states):
        return forward_masked_rnn_transposed(inputs, masks, states, self.inner)
