from common.train import SingleTrainer, AbstractTrainer
from common import AbstractAgent
import numpy as np
import random
import os
from collections import namedtuple
import abc

from .replay import ReplayBuffer
import torch

class DeepQTrainer(SingleTrainer):
    def __init__(self, name, env_kwargs, model_kwargs):
        super().__init__(env_kwargs = env_kwargs, model_kwargs = model_kwargs)  
        self.name = name     
        self.minibatch_size = 32
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

        self._global_t = 0
        self._state = None
        self._episode_length = 0
        self._episode_reward = 0.0
        self._local_timestep = 0
        self._replay = None

    @abc.abstractclassmethod
    def create_model(self, **kwargs):
        pass


    def _build_training(self, model, main_device):
        def compute_loss(observations, actions, rewards, next_observations, terminals):
            with torch.no_grad():
                q_next_online_net = model(next_observations
                
            pcontinues = (1.0 - terminals) * self.gamma

            


    def _build_model_for_training(self):
        inputs = self.create_inputs("main", **self.model_kwargs)
        model = self.create_model(inputs, **self.model_kwargs)
        model_vars = model.trainable_weights
        q = model.output        

        with tf.name_scope('training'):
            # Input placeholders
            actions = tf.placeholder(tf.int32, (None,), name="action")
            rewards = tf.placeholder(tf.float32, (None,), name="reward")
            inputs_next = self.create_inputs("next", **self.model_kwargs)
            terminates = tf.placeholder(tf.bool, (None,), name="terminate")


            q_next_online_net = tf.stop_gradient(model(inputs_next))
            pcontinues = (1.0 - tf.to_float(terminates)) * self.gamma

            
            if self.double_dqn:  
                # Target network      
                target_model = self.create_model(inputs_next, **self.model_kwargs)
                target_vars = target_model.trainable_weights

                q_next = tf.stop_gradient(target_model.output)                        
                errors, _info = double_qlearning(q, actions, rewards, pcontinues, q_next, q_next_online_net)
            else:
                errors, _info = qlearning(q, actions, rewards, pcontinues, q_next_online_net)

            td_error = _info.td_error

            loss = K.mean(errors)
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            optimize_expr = optimizer.minimize(loss, var_list=model_vars)

            with tf.control_dependencies([optimize_expr]):
                optimize_expr = tf.group(*[tf.assign(*a) for a in model.updates])

            if self.double_dqn:
                # update_target_fn will be called periodically to copy Q network to target Q network
                update_target_expr = tf.group(*[var_target.assign(var) for var, var_target in zip(model_vars, target_vars)])
            else:
                update_target_expr = tf.no_op()

        # Create callable functions
        train_fn = K.function(inputs + [
                actions,
                rewards,                                 
                terminates,
            ] + inputs_next,
            outputs=[td_error],
            updates=[optimize_expr]
        )

        act_fn = K.function(inputs = inputs, outputs = [K.argmax(q, axis = 1)])
        q_fn = K.function(inputs = inputs, outputs = [q])
        update_fn = K.function([], [], updates=[update_target_expr])

        self._update_parameters = lambda: update_fn([])
        self._train = train_fn
        self._act = lambda x: act_fn([x])[0]
        self._q = lambda x: q_fn([x])[0]
        return model


    def _initialize(self, **model_kwargs):
        self._replay = ReplayBuffer(self.replay_size)

        sess = K.get_session()
        model = self._build_model_for_training()

        tf.global_variables_initializer().run(session = sess)
        model.summary()
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
        state, action, reward, next_state, done = self._replay.sample(self.minibatch_size)
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

    def _load(self, checkpoint_dir):
        with open(os.path.join(checkpoint_dir, '%s-model.json' % self.name), 'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(os.path.join(checkpoint_dir, '%s-weights.h5' % self.name))

    def wrap_env(self, env):
        return env

    def q(self, state):
        return self.model.predict([state[None]])[0]

    def act(self, state):
        return np.argmax(self.model.predict([state[None]])[0])