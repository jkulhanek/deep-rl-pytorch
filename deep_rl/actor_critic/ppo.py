from abc import abstractclassmethod
import tempfile
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from ..core import SingleTrainer
from ..common.env import VecTransposeImage, make_vec_envs
from ..common import MetricContext
from ..common.torchsummary import minimal_summary, get_observation_shape
from ..utils import pytorch_call, KeepTensor, detach_all
from ..utils import expand_time_dimension, get_batch_size, split_batches
from ..utils import batch_observations


class RolloutStorage:
    def __init__(self, initial_observations, initial_states=[]):
        self.num_processes = get_batch_size(initial_observations)

        self._terminals = self._last_terminals = np.zeros(shape=(self.num_processes,), dtype=np.bool)
        self._states = self._last_states = initial_states
        self._observations = self._last_observations = initial_observations

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
    def observations(self):
        return self._transform_observation(self._observations)

    @property
    def terminals(self):
        return self._terminals.astype(np.float32)

    @property
    def masks(self):
        return 1 - self.terminals

    @property
    def states(self):
        return self._states

    def insert(self, observations, actions, rewards, terminals, values, action_log_probs, states):
        self._batch.append((self._observations, actions, values, rewards, action_log_probs, terminals))
        self._observations = observations
        self._terminals = terminals
        self._states = states

    def batch(self, last_values, gamma, gae_lambda):
        # Batch in time dimension
        b_actions, b_values, b_rewards, b_action_log_probs, b_terminals = [np.stack([b[i] for b in self._batch], axis=1) for i in range(1, 6)]
        b_observations = batch_observations([o[0] for o in self._batch])

        # Compute cummulative returns
        b_values = np.concatenate([b_values, np.expand_dims(last_values, 1)], axis=1)
        last_returns = (1.0 - b_terminals[:, -1]) * last_values
        b_returns = np.concatenate([np.zeros_like(b_rewards), np.expand_dims(last_returns, 1)], axis=1)
        gae = np.zeros_like(last_values)
        for n in reversed(range(len(self._batch))):
            diff = b_rewards[:, n] + \
                gamma * (1.0 - b_terminals[:, n]) * b_values[:, n + 1] -\
                b_values[:, n]

            gae = diff + gamma * gae_lambda * (1.0 - b_terminals[:, n]) * gae
            b_returns[:, n] = gae + b_values[:, n]

        # Compute RNN reset masks
        b_masks = (1 - np.concatenate([b_terminals[:, 1:], np.expand_dims(self._last_terminals, 1)], axis=1))
        result = (
            self._transform_observation(b_observations),
            b_returns[:, :-1].astype(np.float32),
            b_actions,
            b_masks.astype(np.float32),
            b_values[:, :-1],
            b_action_log_probs,
            self._last_states
        )

        self._last_observations = self._observations
        self._last_states = self._states
        self._last_terminals = self._terminals
        self._batch = []
        return result


class PPO(SingleTrainer):
    def __init__(self, name, env_kwargs, model_kwargs, max_time_steps, **kwargs):
        super().__init__(env_kwargs=env_kwargs, model_kwargs=model_kwargs)

        self.entropy_coefficient = 0.01
        self.value_coefficient = 0.25
        self.max_gradient_norm = 0.5
        self.data_parallel = True
        self.learning_rate = 2e-4
        self.clip_param = 0.1
        self.ppo_epochs = 4
        self.num_minibatches = 8
        self.gae_lambda = 0.95
        self.max_time_steps = max_time_steps
        self.name = name
        self.num_steps = 5
        self.num_processes = 16
        self.gamma = 0.99
        self.allow_gpu = True

        self.log_dir = None
        self.win = None

        def not_initialized(*args, **kwargs):
            raise Exception('Not initialized')
        self._train = self._step = self._value = not_initialized

    def _initialize(self, **model_kwargs):
        model = self._build_graph(self.allow_gpu, **model_kwargs)
        self._tstart = time.time()
        self.rollouts = RolloutStorage(self.env.reset(), self._initial_states(self.num_processes))
        return model

    def save(self, path):
        super().save(path)
        self._save(path)

    def _finalize(self):
        if self.log_dir is not None:
            self.log_dir.cleanup()

    def create_env(self, env):
        self.log_dir = tempfile.TemporaryDirectory()

        seed = 1
        self.validation_env = make_vec_envs(env, seed, 1, self.gamma, self.log_dir.name, None, allow_early_resets=True)
        if len(self.validation_env.observation_space.shape) == 4:
            self.validation_env = VecTransposeImage(self.validation_env)

        envs = make_vec_envs(env, seed + 1, self.num_processes,
                             self.gamma, self.log_dir.name, None, False)

        if len(envs.observation_space.shape) == 4:
            envs = VecTransposeImage(envs)

        return envs

    def process(self, context, mode='train', **kwargs):
        metric_context = MetricContext()
        if mode == 'train':
            return self._process_train(context, metric_context)
        elif mode == 'validation':
            return self._process_validation(metric_context)
        else:
            raise Exception('Mode not supported')

    def _process_validation(self, metric_context):
        done = False
        states = self._initial_states(1)
        ep_reward = 0.0
        ep_length = 0
        n_steps = 0
        observations = self.validation_env.reset()
        while not done:
            action, _, _, states = self._step(observations, np.ones((1, 1), dtype=np.float32), states)
            observations, reward, done, infos = self.validation_env.step(action)
            done = done[0]
            info = infos[0]

            if 'episode' in info.keys():
                ep_length = info['episode']['l']
                ep_reward = info['episode']['r']
            n_steps += 1

        return n_steps, (ep_length, ep_reward), metric_context

    def _sample_experience_batch(self):
        finished_episodes = ([], [])
        for _ in range(self.num_steps):
            actions, values, action_log_prob, states = self._step(self.rollouts.observations, self.rollouts.masks, self.rollouts.states)

            # Take actions in env and look the results
            observations, rewards, terminals, infos = self.env.step(actions)

            # Collect true rewards
            for info in infos:
                if 'episode' in info.keys():
                    finished_episodes[0].append(info['episode']['l'])
                    finished_episodes[1].append(info['episode']['r'])

            self.rollouts.insert(observations, actions, rewards, terminals, values, action_log_prob, states)

        last_values, _ = self._value(self.rollouts.observations, self.rollouts.masks, self.rollouts.states)
        batched = self.rollouts.batch(last_values, self.gamma, self.gae_lambda)

        # Prepare next batch starting point
        return batched, (len(finished_episodes[0]),) + finished_episodes

    def _process_train(self, context, metric_context):
        batch, report = self._sample_experience_batch()
        loss, value_loss, action_loss, dist_entropy = self._train(*batch)

        fps = int(self._global_t / (time.time() - self._tstart))
        metric_context.add_cummulative('updates', 1)
        metric_context.add_scalar('loss', loss)
        metric_context.add_scalar('value_loss', value_loss)
        metric_context.add_scalar('action_loss', action_loss)
        metric_context.add_scalar('entropy', dist_entropy)
        metric_context.add_last_value_scalar('fps', fps)
        return self.num_steps * self.num_processes, report, metric_context

    @abstractclassmethod
    def create_model(self, **kwargs):
        pass

    def show_summary(self, model):
        batch_shape = (self.num_processes, self.num_steps)

        def get_shape_rec(shapes):
            if isinstance(shapes, tuple):
                return tuple(get_shape_rec(list(shapes)))
            elif isinstance(shapes, list):
                return [get_shape_rec(x) for x in shapes]
            else:
                return shapes.size()

        shapes = tuple(get_observation_shape(batch_shape, self.env.observation_space)) + (batch_shape, get_shape_rec(self._initial_states(self.num_processes)))
        minimal_summary(model, shapes)

    def _build_train(self, model, main_device):
        optimizer = optim.Adam(model.parameters(), self.learning_rate)

        @pytorch_call(main_device)
        def train_minibatch(observations, returns, actions, masks, old_value_preds, old_action_log_probs, states, advantages):
            policy_logits, value, _ = model(observations, masks, states)
            value = value.view(value.shape[:-1])

            dist = torch.distributions.Categorical(logits=policy_logits)
            action_log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy().mean()

            ratio = torch.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                1.0 + self.clip_param) * advantages
            action_loss = -torch.min(surr1, surr2).mean()

            value_pred_clipped = old_value_preds + \
                (value - old_value_preds).clamp(-self.clip_param, self.clip_param)
            value_loss = (value - returns).pow(2)
            value_loss_clipped = (value_pred_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss,
                                         value_loss_clipped).mean()

            # Compute losses
            loss = value_loss * self.value_coefficient + \
                action_loss - \
                dist_entropy * self.entropy_coefficient

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.max_gradient_norm)
            optimizer.step()

            return loss.item(), action_loss.item(), value_loss.item(), dist_entropy.item()

        def train(*args):
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate

            # observations, returns, actions, masks, old_value_preds
            loss, action_loss, value_loss, dist_entropy = 0, 0, 0, 0
            advantages = args[1] - args[4]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            for e in range(self.ppo_epochs):
                generator = split_batches(self.num_minibatches, args + (advantages,))
                for x in generator:
                    l, al, vl, de = train_minibatch(*x)
                    loss += l
                    action_loss += al
                    value_loss += vl
                    dist_entropy += de
            num_updates = self.ppo_epochs * self.num_minibatches
            return loss / num_updates, action_loss / num_updates, \
                value_loss / num_updates, dist_entropy / num_updates
        return train

    def _build_graph(self, allow_gpu, **model_kwargs):
        model = self.create_model(**model_kwargs)
        if hasattr(model, 'initial_states'):
            self._initial_states = getattr(model, 'initial_states')
        else:
            self._initial_states = lambda _: []

        # Show summary
        self.show_summary(model)

        cuda_devices = torch.cuda.device_count()
        if cuda_devices == 0 or not allow_gpu:
            print('Using CPU only')
            main_device = torch.device('cpu')
            def get_state_dict(): return model.state_dict()
        elif cuda_devices > 1 and self.data_parallel:
            print('Using %s GPUs' % cuda_devices)
            main_device = torch.device('cuda:0')
            model = nn.DataParallel(model, output_device=main_device)
            model = model.to(main_device)
            def get_state_dict(): return model.module.state_dict()
        else:
            print('Using single GPU')
            main_device = torch.device('cuda:0')
            model = model.to(main_device)
            def get_state_dict(): return model.state_dict()

        model.train()

        # Build train and act functions
        self._train = self._build_train(model, main_device)

        @pytorch_call(main_device)
        def step(observations, masks, states):
            with torch.no_grad():
                batch_size = get_batch_size(observations)
                observations = expand_time_dimension(observations)
                masks = masks.view(batch_size, 1)

                policy_logits, value, states = model(observations, masks, states)
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample()
                action_log_probs = dist.log_prob(action)
                return action.squeeze(1).detach(), value.squeeze(1).squeeze(-1).detach(), action_log_probs.squeeze(1).detach(), KeepTensor(detach_all(states))

        @pytorch_call(main_device)
        def value(observations, masks, states):
            with torch.no_grad():
                batch_size = get_batch_size(observations)
                observations = expand_time_dimension(observations)
                masks = masks.view(batch_size, 1)

                _, value, states = model(observations, masks, states)
                return value.squeeze(1).squeeze(-1).detach(), KeepTensor(detach_all(states))

        self._step = step
        self._value = value
        self._save = lambda path: torch.save(get_state_dict(), os.path.join(path, 'weights.pth'))
        self.main_device = main_device
        return model
