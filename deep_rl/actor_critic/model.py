import torch
import torch.nn as nn
import math

from deep_rl.utils.model import TimeDistributed, Flatten
from ..common.pytorch import forward_masked_rnn_transposed


class CNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        def init_layer(layer, activation=None, gain=None):
            if activation is not None and gain is None:
                gain = nn.init.calculate_gain(activation.lower())
            elif activation is None and gain is None:
                gain = 1.0

            nn.init.orthogonal_(layer.weight.data, gain=gain)
            nn.init.zeros_(layer.bias.data)
            output = [layer]
            if activation is not None:
                output.append(getattr(nn, activation)())
            return output

        layers = []
        layers.extend(init_layer(nn.Conv2d(num_inputs, 32, 8, stride=4), activation='ReLU'))
        layers.extend(init_layer(nn.Conv2d(32, 64, 4, stride=2), activation='ReLU'))
        layers.extend(init_layer(nn.Conv2d(64, 32, 3, stride=1), activation='ReLU'))
        layers.append(Flatten())
        layers.extend(init_layer(nn.Linear(32 * 7 * 7, 512), activation='ReLU'))

        self.main = nn.Sequential(*layers)
        self.critic = init_layer(nn.Linear(512, 1))[0]
        self.policy_logits = init_layer(nn.Linear(512, num_outputs), gain=0.01)[0]

    def forward(self, inputs):
        main_features = self.main(inputs)
        policy_logits = self.policy_logits(main_features)
        critic = self.critic(main_features)
        return policy_logits, critic

    @property
    def output_names(self):
        return ('policy_logits', 'value', 'states')


def TimeDistributedCNN(num_inputs, num_outputs):
    inner = CNN(num_inputs, num_outputs)
    model = TimeDistributed(inner)
    model.output_names = property(lambda self: inner.output_names)
    _forward = model.forward
    model.forward = lambda inputs, masks, states: _forward(inputs) + (states, )
    return model


class TimeDistributedModel(nn.Module):
    def __init__(self):
        super().__init__()

    def init_layer(self, layer, activation=None, gain=None):
        if activation is not None and gain is None:
            gain = nn.init.calculate_gain(activation.lower())
        elif activation is None and gain is None:
            gain = 1.0

        nn.init.orthogonal_(layer.weight.data, gain=gain)
        nn.init.zeros_(layer.bias.data)
        output = [TimeDistributed(layer)]
        if activation is not None:
            output.append(TimeDistributed(getattr(nn, activation)()))
        return output

    def init_nondistributed_layer(self, layer, activation=None, gain=None):
        if activation is not None and gain is None:
            gain = nn.init.calculate_gain(activation.lower())
        elif activation is None and gain is None:
            gain = 1.0

        nn.init.orthogonal_(layer.weight.data, gain=gain)
        nn.init.zeros_(layer.bias.data)
        output = [layer]
        if activation is not None:
            output.append(getattr(nn, activation)())
        return output


class TimeDistributedConv(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.main_output_size = 512

        def init_layer(layer, activation=None, gain=None):
            if activation is not None and gain is None:
                gain = nn.init.calculate_gain(activation.lower())
            elif activation is None and gain is None:
                gain = 1.0

            nn.init.orthogonal_(layer.weight.data, gain=gain)
            nn.init.zeros_(layer.bias.data)
            output = [TimeDistributed(layer)]
            if activation is not None:
                output.append(TimeDistributed(getattr(nn, activation)()))
            return output

        layers = []
        layers.extend(init_layer(nn.Conv2d(num_inputs, 32, 8, stride=4), activation='ReLU'))
        layers.extend(init_layer(nn.Conv2d(32, 64, 4, stride=2), activation='ReLU'))
        layers.extend(init_layer(nn.Conv2d(64, 32, 3, stride=1), activation='ReLU'))
        layers.append(TimeDistributed(Flatten()))
        layers.extend(init_layer(nn.Linear(32 * 7 * 7, self.main_output_size), activation='ReLU'))
        self.main = nn.Sequential(*layers)

        self.critic = init_layer(nn.Linear(self.main_output_size, 1))[0]
        self.policy_logits = init_layer(nn.Linear(512, num_outputs), gain=0.01)[0]

    def forward(self, inputs, masks, states):
        main_features = self.main(inputs)
        policy_logits = self.policy_logits(main_features)
        critic = self.critic(main_features)
        return policy_logits, critic, states

    @property
    def output_names(self):
        return ('policy_logits', 'value', 'states')


class LSTMConv(TimeDistributedConv):
    def __init__(self, num_inputs, num_outputs):
        super().__init__(num_inputs, num_outputs)

        self.lstm_layers = 1
        self.lstm_hidden_size = 128
        self.rnn = nn.LSTM(self.main_output_size,
                           hidden_size=self.lstm_hidden_size,
                           num_layers=self.lstm_layers,
                           batch_first=True)

    def initial_states(self, batch_size):
        return tuple([torch.zeros([batch_size, self.lstm_layers, self.lstm_hidden_size], dtype=torch.float32) for _ in range(2)])

    def forward(self, inputs, masks, states):
        main_features = self.main(inputs)
        main_features, states = forward_masked_rnn_transposed(main_features, masks, states, self.rnn.forward)

        policy_logits = self.policy_logits(main_features)
        critic = self.critic(main_features)
        return [policy_logits, critic, states]


class TimeDistributedMultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        def init_layer(layer, activation=None, gain=None):
            if activation is not None and gain is None:
                gain = nn.init.calculate_gain(activation.lower())
            elif activation is None and gain is None:
                gain = 1.0

            nn.init.orthogonal_(layer.weight.data, gain=gain)
            nn.init.zeros_(layer.bias.data)
            output = [TimeDistributed(layer)]
            if activation is not None:
                output.append(TimeDistributed(getattr(nn, activation)()))
            return output

        hidden_size = 512
        self.actor = nn.Sequential(*
                                   init_layer(nn.Linear(input_size, hidden_size), activation='Tanh') +
                                   init_layer(nn.Linear(hidden_size, hidden_size), activation='Tanh') +
                                   init_layer(nn.Linear(hidden_size, output_size), activation=None, gain=0.01)
                                   )

        self.critic = nn.Sequential(*
                                    init_layer(nn.Linear(input_size, hidden_size), activation='Tanh') +
                                    init_layer(nn.Linear(hidden_size, hidden_size), activation='Tanh') +
                                    init_layer(nn.Linear(hidden_size, 1), activation=None, gain=1.0)
                                    )

    def forward(self, inputs, masks, states):
        x = inputs
        return self.actor(x), self.critic(x), states


class LSTMMultiLayerPerceptron(TimeDistributedMultiLayerPerceptron):
    def __init__(self, input_size, output_size):
        self.lstm_hidden_size = 64

        super().__init__(self.lstm_hidden_size, output_size)
        self.lstm = nn.LSTM(input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=1,
                            batch_first=True)

    def initial_states(self, batch_size):
        return tuple([torch.zeros([batch_size, 1, self.lstm_hidden_size], dtype=torch.float32) for _ in range(2)])

    def forward(self, inputs, masks, states):
        features = inputs
        features, states = forward_masked_rnn_transposed(features, masks, states, self.lstm.forward)
        return super()(features, masks, states)


class UnrealModel(nn.Module):
    def init_weights(self, module):
        if type(module) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

        elif type(module) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
            nn.init.zeros_(module.bias.data)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight.data)
            d = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(module.weight.data, -d, d)

    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.conv_base = TimeDistributed(nn.Sequential(
            nn.Conv2d(num_inputs, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
        ))

        self.conv_merge = TimeDistributed(nn.Sequential(
            Flatten(),
            nn.Linear(9 ** 2 * 32, 256),
            nn.ReLU()
        ))

        self.main_output_size = 256

        self.critic = TimeDistributed(nn.Linear(self.main_output_size, 1))
        self.policy_logits = TimeDistributed(nn.Linear(self.main_output_size, num_outputs))

        self.lstm_layers = 1
        self.lstm_hidden_size = 256
        self.rnn = nn.LSTM(256 + num_outputs + 1,  # Conv outputs + last action, reward
                           hidden_size=self.lstm_hidden_size,
                           num_layers=self.lstm_layers,
                           batch_first=True)

        self._create_pixel_control_network(num_outputs)
        self._create_rp_network()

        self.pc_cell_size = 4
        self.apply(self.init_weights)

    def initial_states(self, batch_size):
        return tuple([torch.zeros([batch_size, self.lstm_layers, self.lstm_hidden_size], dtype=torch.float32) for _ in range(2)])

    def forward(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        policy_logits = self.policy_logits(features)
        critic = self.critic(features)
        return [policy_logits, critic, states]

    def _forward_base(self, inputs, masks, states):
        observations, last_reward_action = inputs['observation'], inputs['action_reward']
        features = self.conv_base(observations)
        features = self.conv_merge(features)
        features = torch.cat((features, last_reward_action,), dim=2)
        return forward_masked_rnn_transposed(features, masks, states, self.rnn.forward)

    def _create_pixel_control_network(self, num_outputs):
        self.pc_base = TimeDistributed(nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 32 * 9 * 9),
            nn.ReLU()
        ))

        self.pc_action = TimeDistributed(nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2),
            nn.ReLU()
        ))

        self.pc_value = TimeDistributed(nn.Sequential(
            nn.ConvTranspose2d(32, num_outputs, kernel_size=4, stride=2),
            nn.ReLU()
        ))

    def _create_rp_network(self):
        self.rp = nn.Linear(9 ** 2 * 32 * 3, 3)

    def reward_prediction(self, inputs):
        observations = inputs['observation']
        features = self.conv_base(observations)
        features = features.view(features.size()[0], -1)
        features = self.rp(features)
        return features

    def pixel_control(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        features = self.pc_base(features)
        features = features.view(*(features.size()[:2] + (32, 9, 9)))
        action_features = self.pc_action(features)
        features = self.pc_value(features) + action_features - action_features.mean(2, keepdim=True)
        return features, states

    def value_prediction(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        critic = self.critic(features)
        return critic, states
