import pytest
from unittest.mock import patch
import sys


@pytest.mark.parametrize("trainer,max_time_steps", [('paac', 4000), ('ppo', 2000), ('a3c', 16000)])
@pytest.mark.slow
def test_actor_critic_cartpole(trainer, max_time_steps):
    import experiments.actor_critic_cartpole
    with patch.object(sys, 'argv', ['train.py', '--num-eval-episodes', '10',
                                    '--max-time-steps', str(max_time_steps), '--trainer', trainer]):
        m = experiments.actor_critic_cartpole.main()
        assert m['return'] > 30


@pytest.mark.slow
def test_dqn_cartpole():
    import experiments.deepq_cartpole
    max_time_steps = 200000
    with patch.object(sys, 'argv', ['train.py', '--num-eval-episodes', '10',
                                    '--preload-steps', str(int(max_time_steps / 10)),
                                    '--max-time-steps', str(max_time_steps), '--trainer', 'dqn']):
        m = experiments.deepq_cartpole.main()
        assert m['return'] > 30


@pytest.mark.slow
def test_dqn_prioritized_replay_cartpole():
    import experiments.deepq_cartpole
    max_time_steps = 20000
    with patch.object(sys, 'argv', ['train.py', '--num-eval-episodes', '10',
                                    '--prioritized-replay-alpha', '0.5',
                                    '--preload-steps', str(int(max_time_steps / 10)),
                                    '--max-time-steps', str(max_time_steps), '--trainer', 'dqn']):
        m = experiments.deepq_cartpole.main()
        assert m['return'] > 30


@pytest.mark.parametrize("trainer,max_time_steps", [('ppo', 200)])
@pytest.mark.slow
def test_actor_critic_atari(trainer, max_time_steps):
    import experiments.actor_critic_atari
    with patch.object(sys, 'argv', ['train.py', '--num-eval-episodes', '10',
                                    '--replay-size', '40',
                                    '--max-time-steps', str(max_time_steps), '--trainer', trainer]):
        m = experiments.actor_critic_atari.main()
        assert m['return'] >= 0.3


@pytest.mark.parametrize("trainer", ['unreal', 'unreala3c', 'ppo'])
@pytest.mark.slow
def test_actor_critic_unreal(trainer):
    import experiments.actor_critic_atari
    import gym
    with patch.object(sys, 'argv', ['train.py', '--num-eval-episodes', '10',
                                    '--replay-size', '10',
                                    '--num-steps', '5',
                                    '--max-time-steps', '10', '--trainer', trainer]):
        _make = gym.make
        with patch.object(gym, 'make', lambda *args, **kwargs: gym.wrappers.TimeLimit(_make(*args, **kwargs), 20)):
            experiments.actor_critic_atari.main()
