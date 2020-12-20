from abc import abstractclassmethod
import torch
import torch.nn as nn
import numpy as np

from ..common import MetricContext
from ..core import Schedule
from ..utils import pytorch_call, KeepTensor, detach_all
from ..common.multiprocessing import ProcessServerTrainer
from ..common.util import serialize_function
from ..optim.shared_rmsprop import SharedRMSprop


def batch_observations(observations):
    if isinstance(observations[0], tuple):
        return tuple(batch_observations(list(map(list, observations))))
    elif isinstance(observations[0], list):
        return list(map(lambda *x: batch_observations(x), *observations))
    elif isinstance(observations[0], dict):
        return {key: batch_observations([o[key] for o in observations]) for key in observations[0].keys()}
    else:
        return np.stack(observations, axis=0)


def expand_time_dimension(inputs):
    if isinstance(inputs, list):
        return [expand_time_dimension(x) for x in inputs]
    elif isinstance(inputs, tuple):
        return tuple(expand_time_dimension(list(inputs)))
    else:
        return inputs.unsqueeze(0)


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


class RolloutStorage:
    def __init__(self, initial_observation, initial_states=[]):
        self._terminal = self._last_terminal = False
        self._states = self._last_states = initial_states
        self._observation = self._last_observation = initial_observation
        self._batch = []

    def _transform_observation(self, observation):
        if isinstance(observation, np.ndarray):
            if observation.dtype == np.uint8:
                return observation.astype(np.float32) / 255.0
            else:
                return observation.astype(np.float32)
        elif isinstance(observation, list):
            return [self._transform_observation(x) for x in observation]
        elif isinstance(observation, tuple):
            return tuple([self._transform_observation(x) for x in observation])

    @property
    def observation(self):
        return self._transform_observation(self._observation)

    @property
    def terminal(self):
        return float(self._terminal)

    @property
    def mask(self):
        return 1 - self._terminal

    @property
    def states(self):
        return self._states

    def insert(self, observation, action, reward, terminal, value, states):
        self._batch.append((self._observation, action, value, reward, terminal))
        self._observation = observation
        self._terminal = terminal
        self._states = states

    def batch(self, last_value, gamma):
        # Batch in time dimension
        b_actions, b_values, b_rewards, b_terminals = [np.stack([b[i] for b in self._batch], axis=0) for i in range(1, 5)]
        b_observations = batch_observations([o[0] for o in self._batch])

        # Compute cummulative returns
        last_returns = (1.0 - b_terminals[-1]) * last_value
        b_returns = np.concatenate([np.zeros_like(b_rewards), np.array([last_returns], dtype=np.float32)], axis=0)
        for n in reversed(range(len(self._batch))):
            b_returns[n] = b_rewards[n] + \
                gamma * (1.0 - b_terminals[n]) * b_returns[n + 1]

        # Compute RNN reset masks
        b_masks = (1 - np.concatenate([np.array([self._last_terminal], dtype=np.bool), b_terminals[:-1]], axis=0))
        result = (
            self._transform_observation(b_observations),
            b_returns[:-1].astype(np.float32),
            b_actions,
            b_masks.astype(np.float32),
            self._last_states
        )

        self._last_observation = self._observation
        self._last_states = self._states
        self._last_terminal = self._terminal
        self._batch = []
        return result


class A3CWorker:
    def __init__(self, shared_model, optimizer, create_model_fn):
        self.shared_model = shared_model
        self.optimizer = optimizer
        self.create_model_fn = create_model_fn
        self.create_env = None

        self._global_t = None
        self._is_stopped = None

        self.entropy_coefficient = 0.01
        self.value_coefficient = 0.5
        self.max_gradient_norm = 40.0
        self.num_steps = 5
        self.gamma = 0.99
        self.schedules = dict()

    def initialize(self):
        self.env = self.create_env()
        self.model = self.create_model_fn(self)()
        self._build_graph()
        self.rollouts = RolloutStorage(self.env.reset(), self._initial_states(1))

    def process(self, context, mode='train', **kwargs):
        for s in self.schedules.values():
            s.step(self._global_t)
        metric_context = MetricContext()
        if mode == 'train':
            return self._process_train(context, metric_context)
        if mode == 'validation':
            return self._process_validation(context, metric_context)
        else:
            raise Exception('Mode not supported')

    def _sample_experience_batch(self):
        finished_episodes = ([], [])
        for _ in range(self.num_steps):
            states = self.rollouts.states if self.rollouts.mask else self._initial_states(1)
            action, value, action_log_prob, states = self._step(self.rollouts.observation, states)

            # Take actions in env and look the results
            observation, reward, terminal, info = self.env.step(action)

            # Collect true rewards
            if 'episode' in info.keys():
                finished_episodes[0].append(info['episode']['l'])
                finished_episodes[1].append(info['episode']['r'])

            if terminal:
                observation = self.env.reset()

            self.rollouts.insert(observation, action, reward, terminal, value, states)

            if terminal:
                # End after episode env
                # Improves lstm performance
                break

        states = self.rollouts.states if self.rollouts.mask else self._initial_states(1)
        last_values, _ = self._value(self.rollouts.observation, states)
        batched = self.rollouts.batch(last_values, self.gamma)

        # Prepare next batch starting point
        return batched, (len(finished_episodes[0]),) + finished_episodes

    def _process_train(self, context, metric_context):
        self.model.load_state_dict(self.shared_model.state_dict())
        batch, report = self._sample_experience_batch()
        loss, value_loss, action_loss, dist_entropy = self._train(*batch)

        metric_context.add_cummulative('updates', 1)
        metric_context.add_scalar('loss', loss)
        metric_context.add_scalar('value_loss', value_loss)
        metric_context.add_scalar('action_loss', action_loss)
        metric_context.add_scalar('entropy', dist_entropy)
        return self.num_steps, report, metric_context

    def _process_validation(self, context, metric_context):
        self.model.load_state_dict(self.shared_model.state_dict())
        batch, report = self._sample_experience_batch()
        return self.num_steps, report, metric_context

    def _build_graph(self):
        model = self.model
        if hasattr(model, 'initial_states'):
            self._initial_states = getattr(model, 'initial_states')
        else:
            self._initial_states = lambda _: []

        main_device = torch.device('cpu')  # TODO: add GPU support
        model.train()

        # Build train and act functions
        self._train = self._build_train(model, main_device)

        @pytorch_call(main_device)
        def step(observations, states):
            with torch.no_grad():
                observations = expand_time_dimension(observations)
                policy_logits, value, states = model(observations, states)
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample()
                action_log_probs = dist.log_prob(action)
                return action.squeeze(0).item(), value.squeeze(0).squeeze(-1).item(), action_log_probs.squeeze(0).item(), KeepTensor(detach_all(states))

        @pytorch_call(main_device)
        def value(observations, states):
            with torch.no_grad():
                observations = expand_time_dimension(observations)
                _, value, states = model(observations, states)
                return value.squeeze(0).squeeze(-1).detach(), KeepTensor(detach_all(states))

        self._step = step
        self._value = value
        self.main_device = main_device
        return model

    def _build_train(self, model, main_device):
        optimizer = self.optimizer

        @pytorch_call(main_device)
        def train(observations, returns, actions, masks, states=[]):
            if isinstance(self.learning_rate, Schedule):
                lr = self.learning_rate()
                # Update learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            policy_logits, value, _ = model(observations, states)

            dist = torch.distributions.Categorical(logits=policy_logits)
            action_log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy().mean()

            # Compute losses
            advantages = returns - value.squeeze(-1)
            value_loss = advantages.pow(2).mean()
            action_loss = -(advantages.detach() * action_log_probs).mean()
            loss = value_loss * self.value_coefficient + \
                action_loss - \
                dist_entropy * self.entropy_coefficient

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.max_gradient_norm)
            ensure_shared_grads(model, self.shared_model)
            optimizer.step()

            return loss.item(), action_loss.item(), value_loss.item(), dist_entropy.item()

        return train


class A3C(ProcessServerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gamma = 0.99
        self.num_steps = 20
        self.entropy_coefficient = 0.01
        self.value_coefficient = 0.5
        self.max_gradient_norm = 40.0
        self.learning_rate = 0.0001
        self.rms_alpha = 0.99
        self.rms_epsilon = 1e-5
        self.log_dir = None

        self.allow_gpu = True

    @abstractclassmethod
    def create_model(self, **model_kwargs):
        pass

    def _initialize(self, **model_kwargs):
        if hasattr(self, 'schedules') and 'learning_rate' in self.schedules:
            raise Exception('Learning rate schedule is not yet implemented for a3c')

        self.model = self.create_model(**model_kwargs).share_memory()
        self.optimizer = SharedRMSprop(self.model.parameters(), self.learning_rate, self.rms_alpha, self.rms_epsilon).share_memory()
        super()._initialize(**model_kwargs)
        return self.model

    def create_worker(self, id):
        model_fn = serialize_function(self.create_model, **self._model_kwargs)
        worker = A3CWorker(self.model, self.optimizer, model_fn)
        worker.gamma = self.gamma
        worker.num_steps = self.num_steps
        worker.entropy_coefficient = self.entropy_coefficient
        worker.value_coefficient = self.value_coefficient
        worker.max_gradient_norm = self.max_gradient_norm
        worker.allow_gpu = self.allow_gpu
        worker.learning_rate = self.learning_rate
        for name, schedule in self.schedules.items():
            setattr(worker, name, schedule)
        worker.schedules = self.schedules
        return worker
