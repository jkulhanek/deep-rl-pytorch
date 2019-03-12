import torch
import torch.nn as nn

from ..a2c.model import TimeDistributedModel, TimeDistributed, Flatten
from ..common.pytorch import forward_masked_rnn_transposed

class UnrealModel(TimeDistributedModel):
    def init_weights(self, module):
        if type(module) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        layers = []
        layers.extend(self.init_layer(nn.Conv2d(num_inputs, 16, 8, stride = 4), activation='ReLU'))
        layers.extend(self.init_layer(nn.Conv2d(16, 32, 4, stride = 2), activation='ReLU'))
        layers.append(TimeDistributed(Flatten()))
        layers.extend(self.init_layer(nn.Linear(9 ** 2 * 32, 256), activation='ReLU'))
        self.conv = nn.Sequential(*layers)
        self.main_output_size = 256
        
        self.critic = self.init_layer(nn.Linear(self.main_output_size, 1))[0]
        self.policy_logits = self.init_layer(nn.Linear(self.main_output_size, num_outputs), gain = 0.01)[0]

        self.lstm_layers = 1
        self.lstm_hidden_size = 256
        self.rnn = nn.LSTM(256 + num_outputs + 1, # Conv outputs + last action, reward
            hidden_size = self.lstm_hidden_size, 
            num_layers = self.lstm_layers,
            batch_first = True)
        self.rnn.apply(self.init_weights)

        self._create_pixel_control_network(num_outputs)
        self._create_rp_network()

    def initial_states(self, batch_size):
        return tuple([torch.zeros([batch_size, self.lstm_layers, self.lstm_hidden_size], dtype = torch.float32) for _ in range(2)])

    def forward(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        policy_logits = self.policy_logits(features)
        critic = self.critic(features)
        return [policy_logits, critic, states]

    def _forward_base(self, inputs, masks, states):
        observations, last_reward_action = inputs
        conv_features = self.conv(observations)
        features = torch.cat((conv_features, last_reward_action,), dim = 2)
        return forward_masked_rnn_transposed(features, masks, states, self.rnn.forward)

    def _create_pixel_control_network(self, num_outputs):
        self.pc_base = nn.Sequential(*self.init_layer(nn.Linear(self.lstm_hidden_size, 32 * 9 * 9), activation='ReLU'))
        self.pc_action = self.init_layer(nn.ConvTranspose2d(32, 1, kernel_size = 4, stride = 2), activation=None)[0] # TODO: try ReLU as in original research
        self.pc_value = self.init_layer(nn.ConvTranspose2d(32, num_outputs, kernel_size = 4, stride=2), activation=None)[0] # TODO: try ReLU as in original research

    def _create_rp_network(self):
        self.rp = self.init_nondistributed_layer(nn.Linear(9 ** 2 * 32 * 3, 3), activation=None)[0]

    def reward_prediction(self, inputs):
        observations, _ = inputs
        features = observations.view(observations.size()[0], -1)
        features = self.rp(features)
        return features

    def pixel_control(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        features = self.pc_base(features)
        action_features = self.pc_action(features)
        features = self.pc_value(features) + action_features - action_features.mean(-1)
        return features, states

    def value_prediction(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        critic = self.critic(features)
        return critic, states

    @property
    def output_names(self):
        return ('policy_logits', 'value', 'states')