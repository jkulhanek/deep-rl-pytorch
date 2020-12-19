from abc import abstractclassmethod
import torch
import os
from ..core import AbstractAgent
from ..configuration import configuration
from ..utils import expand_time_and_batch_dimensions, KeepTensor, detach_all, pytorch_call


def wrap_agent_env(thunk):
    from ..common.env import ScaledFloatFrame, TransposeImage

    def _thunk():
        env = thunk()
        env = TransposeImage(env)
        env = ScaledFloatFrame(env)
        return env
    return _thunk


class ActorCriticAgent(AbstractAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = self._initialize()
        self.states = None

    @abstractclassmethod
    def create_model(self):
        pass

    def _initialize(self):
        checkpoint_dir = configuration.get('models_path')

        path = os.path.join(checkpoint_dir, self.name, 'weights.pth')
        model = self.create_model()
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()

        @pytorch_call(torch.device('cpu'))
        def step(observations, states):
            with torch.no_grad():
                observations = expand_time_and_batch_dimensions(observations)
                masks = torch.ones((1, 1), dtype=torch.float32)

                policy_logits, _, states = model(observations, masks, states)
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample()
                return action.item(), KeepTensor(detach_all(states))

        self._step = step
        return model

    def wrap_env(self, env):
        def _thunk():
            return env

        return wrap_agent_env(_thunk)()

    def reset_state(self):
        self.states = None

    def act(self, state):
        action, self.states = self._step(state, self.states)
        return action
