import torch
from torch import nn
from torch.nn import functional as F
import math
from deep_rl.utils.model import Flatten


class NoisyFactorizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        # Initialize
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
        self.reset_noise()

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


class RainbowBasedModel(nn.Module):
    def __init__(self, noisy_nets: bool = True, distributional: bool = True, distributional_atoms: int = 51):
        super().__init__()
        self.noisy_nets = noisy_nets
        self.distributional = distributional
        self.distributional_atoms = distributional_atoms if distributional else None
        self.Linear = NoisyFactorizedLinear if noisy_nets else nn.Linear
        self.output_multiple = distributional_atoms if distributional else 1
        if noisy_nets:
            def reset_noise():
                def reset_layer(x):
                    if x != self and hasattr(x, 'reset_noise'):
                        x.reset_noise()
                self.apply(reset_layer)
            self.reset_noise = reset_noise


class RainbowConvModel(RainbowBasedModel):
    def __init__(self, n_actions, noisy_nets: bool = True, distributional: bool = True, distributional_atoms: int = 51):
        super().__init__(noisy_nets=noisy_nets, distributional=distributional, distributional_atoms=distributional_atoms)
        self.n_actions = n_actions
        self.trunk = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            Flatten()
        )

        trunk_output = 7 * 7 * 64
        self.policy = nn.Sequential(
            self.Linear(trunk_output, 512),
            nn.ReLU(),
            self.Linear(512, n_actions * self.output_multiple)
        )
        self.value = nn.Sequential(
            self.Linear(trunk_output, 512),
            nn.ReLU(),
            self.Linear(512, self.output_multiple),
        )

    def forward(self, x):
        x = self.trunk(x)
        adventage = self.policy(x)
        value = self.value(x)
        if self.distributional_atoms is not None:
            adventage = adventage.view(*(adventage.shape[:-1] + (self.n_actions, self.distributional_atoms,)))
            value = value.view(*(value.shape[:-1] + (1, self.distributional_atoms,)))
            return adventage + value - adventage.mean(-2, keepdim=True)
        else:
            return adventage + value - adventage.mean(-1, keepdim=True)
