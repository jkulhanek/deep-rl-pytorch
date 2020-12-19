from deep_rl import register_trainer
from deep_rl.actor_critic import A2C
from deep_rl.actor_critic.model import TimeDistributedConv


@register_trainer(max_time_steps=10e6, validation_period=None,  episode_log_interval=10, save=False)
class Trainer(A2C):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 16
        self.num_steps = 5
        self.gamma = .99

    def create_model(self):
        return TimeDistributedConv(self.env.single_observation_space.shape[0], self.env.single_action_space.n)


def default_args():
    return dict(
        env_kwargs='BreakoutNoFrameskip-v4',
        model_kwargs=dict()
    )
