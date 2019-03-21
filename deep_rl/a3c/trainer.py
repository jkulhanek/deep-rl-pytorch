from abc import abstractclassmethod
from collections import namedtuple
import tempfile
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..core import AbstractTrainer, SingleTrainer, AbstractAgent
from ..common import MetricContext
from ..common.torchsummary import minimal_summary

from ..a2c.model import TimeDistributedConv
from ..common.pytorch import pytorch_call, to_tensor, to_numpy, KeepTensor, detach_all

from .storage import RolloutStorage

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def expand_time_dimension(inputs):
    if isinstance(inputs, list):
        return [expand_time_dimension(x) for x in inputs]
    elif isinstance(inputs, tuple):
        return tuple(expand_time_dimension(list(inputs)))
    else:
        return inputs.unsqueeze(0)

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

    def run(self, process, **kwargs):
        self._global_t = 0
        self._is_stopped = False
        while not self._is_stopped:
            tdiff, _, _ = process(mode = 'train', context = dict())
            self._global_t += tdiff

        return None


    def initialize(self):
        self.env = self.create_env()
        self.model = self.create_model_fn(self)()
        self._build_graph()
        self.rollouts = RolloutStorage(self.env.reset(), self._initial_states(1))

    def process(self, context, mode = 'train', **kwargs):
        metric_context = MetricContext()
        if mode == 'train':
            return self._process_train(context, metric_context)
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

    def _build_graph(self):
        model = self.model
        if hasattr(model, 'initial_states'):
            self._initial_states = getattr(model, 'initial_states')
        else:
            self._initial_states = lambda _: []

        main_device = torch.device('cpu')
        model.train()

        # Build train and act functions
        self._train = self._build_train(model, main_device)

        @pytorch_call(main_device)
        def step(observations, states):
            with torch.no_grad():
                observations = expand_time_dimension(observations)
                policy_logits, value, states = model(observations, states)
                dist = torch.distributions.Categorical(logits = policy_logits)
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
        def train(observations, returns, actions, masks, states = []):
            policy_logits, value, _ = model(observations, states)

            dist = torch.distributions.Categorical(logits = policy_logits)
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