from common.train import SingleTrainer, AbstractTrainer
from common import AbstractAgent
import numpy as np
import random
import os
from collections import namedtuple
import abc

from .replay import ReplayBuffer
from .util import qlearning, polyak_update, double_qlearning
import torch
from torch import optim
from util.pytorch import pytorch_call
from common.torchsummary import minimal_summary


class DeepQTrainer(SingleTrainer):
    def __init__(self, name, env_kwargs, model_kwargs):
        super().__init__(env_kwargs = env_kwargs, model_kwargs = model_kwargs)  
        self.name = name     
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.02
        
        self.update_period = 500 # Double DQN update period
        self.annealing_steps = 200000
        self.preprocess_steps = 100000
        self.replay_size = 50000
        self.learning_rate = 0.001
        self.model_kwargs = model_kwargs
        self.max_episode_steps = None  
        self.double_dqn = True
        self.allow_gpu = True

        self._global_t = 0
        self._state = None
        self._episode_length = 0
        self._episode_reward = 0.0
        self._local_timestep = 0
        self._replay = None

    @abc.abstractclassmethod
    def create_model(self, **kwargs):
        pass

    def show_summary(self, model):
        batch_shape = (self.batch_size,)
        shapes = (batch_shape + self.env.observation_space.shape, batch_shape, batch_shape, batch_shape + self.env.observation_space.shape, batch_shape)
        minimal_summary(model, shapes)

    def _build_train(self, model, main_device):
        optimizer = optim.Adam(model.parameters(), lr = self.learning_rate)
        update_params = lambda: None
        if self.double_dqn:
            target_net = self.create_model(self._model_kwargs)
            target_net = target_net.to(main_device)
            update_params = lambda: polyak_update(1.0, target_net, model)

        def compute_loss(observations, actions, rewards, next_observations, terminals):
            with torch.no_grad():
                q_next_online_net = model(next_observations)
                if self.double_dqn:
                    q_select = target_net(next_observations)
                
            q = model(observations)
            pcontinues = (1.0 - terminals) * self.gamma

            if self.double_dqn:
                loss = double_qlearning(q, actions, rewards, pcontinues, q_next_online_net, q_select)
            else:
                loss = qlearning(q, actions, rewards, pcontinues, q_next_online_net)
            return loss

        @pytorch_call(main_device)
        def train(*args):            
            loss = compute_loss(*args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.item()

        self._update_parameters = update_params
        return train


    def _build_graph(self, allow_gpu, **model_kwargs):
        model = self.create_model(**model_kwargs)

        # Show summary
        self.show_summary(model)
        cuda_devices = torch.cuda.device_count()
        if cuda_devices == 0 or not allow_gpu:
            print('Using CPU only')
            main_device = torch.device('cpu')
            get_state_dict = lambda: model.state_dict()
        else:
            print('Using single GPU')
            main_device = torch.device('cuda')
            model = model.to(main_device)
            get_state_dict = lambda: model.state_dict()

        model.train()

        # Build train and act functions
        self._train = self._build_train(model, main_device)

        @pytorch_call(main_device)
        def act(observations):
            observations = observations.unsqueeze(0)
            with torch.no_grad():
                q = model(observations)
                actions = torch.argmax(q, -1)
            return actions.squeeze(0).item()

        @pytorch_call(main_device)
        def q(observations):
            observations = observations.unsqueeze(0)
            with torch.no_grad():
                q = model(observations)
            return q.squeeze(0).detach()


        def save(path):
            torch.save(get_state_dict(), os.path.join(path, 'weights.pth'))

        self._act = act
        self._q = q
        self.save = save
        self.main_device = main_device
        return model

    def _initialize(self, **model_kwargs):
        self._replay = ReplayBuffer(self.replay_size)
        model = self._build_graph(self.allow_gpu, **model_kwargs)
        return model

    @property
    def epsilon(self):
        start_eps = self.epsilon_start
        end_eps = self.epsilon_end
        if self._global_t < self.preprocess_steps:
            return 1.0

        return max(start_eps - (start_eps - end_eps) * ((self._global_t - self.preprocess_steps) / self.annealing_steps), end_eps)

    def step(self, state, mode = 'validation'):
        if random.random() < self.epsilon:
            return random.randrange(self.model_kwargs.get('action_space_size'))

        return self.act(state)

    def on_optimization_done(self, stats):
        '''
        Callback called after the optimization step
        '''
        pass

    def act(self, state):
        return self._act(state[None])[0]

    def _optimize(self):
        state, action, reward, next_state, done = self._replay.sample(self.batch_size)
        td_losses = self._train([state, action, reward, done, next_state])
        loss = np.mean(np.abs(td_losses))
        if self._global_t % self.update_period == 0:
            self._update_parameters()

        return loss

    def process(self, mode = 'train', **kwargs):
        episode_end = None

        if self._state is None:
            self._state = self.env.reset()

        old_state = self._state
        action = self.step(self._state, mode)
        self._state, reward, done, env_props = self.env.step(action)

        if mode == 'train':
            self._replay.add(old_state, action, reward, self._state, done)

        self._episode_length += 1
        self._episode_reward += reward

        if done or (self.max_episode_steps is not None and self._episode_length >= self.max_episode_steps):
            episode_end = (self._episode_length, self._episode_reward)
            self._episode_length = 0
            self._episode_reward = 0.0
            self._state = self.env.reset()

        stats = dict()
        if self._global_t >= self.preprocess_steps and mode == 'train':
            loss = self._optimize()
            stats = dict(loss = loss, epsilon = self.epsilon)
            self.on_optimization_done(stats)

        if 'win' in env_props:
            stats['win'] = env_props['win']
        
        if mode == 'train':
            self._global_t += 1
        return (1, episode_end, stats)

class DeepQAgent(AbstractAgent):
    def __init__(self, checkpoint_dir = './checkpoints', name = 'deepq'):
        super().__init__(name)

        self._load(checkpoint_dir)

    @abc.abstractclassmethod
    def create_model(self, **model_kwargs):
        pass

    def _load(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, self.name, 'weights.pth')
        self.model = self.create_model()
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.model.eval()

    def wrap_env(self, env):
        return env

    def q(self, state):
        with torch.no_grad():
            observation = torch.from_numpy(state).unsqueeze(0)
            return self.model(observation).squeeze(0).numpy()

    def act(self, state):
        with torch.no_grad():
            observation = torch.from_numpy(state).unsqueeze(0)
            action = torch.argmax(self.model(observation), dim = -1).squeeze(0)
            return action.item()