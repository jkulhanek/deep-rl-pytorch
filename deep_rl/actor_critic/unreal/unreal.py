import logging
import torch
from typing import Any
from dataclasses import dataclass

from deep_rl.common.schedules import LinearSchedule
from deep_rl.utils.tensor import to_device
from deep_rl.utils.environment import with_wrapper
from .. import PAAC
from .utils import pixel_control_loss, value_loss, reward_prediction_loss
from .storage import BatchExperienceReplay
from .utils import UnrealEnvWrapper
from deep_rl.utils import tensor as tensor_utils
from functools import partial


@dataclass
class UnrealBatch:
    base: Any
    pixel_control: Any
    reward_prediction: Any
    value_replay: Any

    def to(self, device):
        return to_device(self, device)


without_last_item = partial(tensor_utils.tensor_map, lambda x: x[:, :-1].contiguous())


def build_unreal(BaseClass,
                 learning_rate=LinearSchedule(7e-4, 1e-10, None),
                 pc_gamma=0.9,
                 pc_weight=0.05,
                 vr_weight=1.0,
                 rp_weight=1.0,
                 replay_size=2000,
                 num_steps=20):
    class Unreal(BaseClass):
        def __init__(self, model_fn, env_fn, *args,
                     pc_gamma: float = pc_gamma,
                     pc_weight: float = pc_weight,
                     vr_weight: float = vr_weight,
                     rp_weight: float = rp_weight,
                     num_steps: int = num_steps,
                     replay_size: int = replay_size,
                     **kwargs):
            env_fn = with_wrapper(env_fn, UnrealEnvWrapper)
            super().__init__(model_fn, env_fn, *args, num_steps=num_steps, **kwargs)
            self.pc_gamma = pc_gamma
            self.pc_weight = pc_weight
            self.vr_weight = vr_weight
            self.rp_weight = rp_weight
            self.replay_size = replay_size

        def compute_loss(self, batch: UnrealBatch):
            loss, losses = super().compute_loss(batch.base)

            # Compute auxiliary losses
            if batch.pixel_control is not None:
                pixel_control_loss = self._loss_pixel_control(self.model, batch.pixel_control, self.current_device)
                loss += pixel_control_loss * self.pc_weight
                losses['pc_loss'] = pixel_control_loss

            # Compute value replay gradients
            if batch.value_replay is not None:
                value_replay_loss = self._loss_value_replay(self.model, batch.value_replay, self.current_device)
                loss += value_replay_loss * self.vr_weight
                losses['vr_loss'] = value_replay_loss

            # Compute reward prediction gradients
            if batch.reward_prediction is not None:
                reward_prediction_loss = self._loss_reward_prediction(self.model, batch.reward_prediction)
                loss += reward_prediction_loss * self.rp_weight
                losses['rp_loss'] = reward_prediction_loss
            return loss, losses

        def sample_experience_batch(self):
            batch = super().sample_experience_batch()
            pc_batch = self.replay.sample_sequence() if self.pc_weight > 0.0 else None
            vr_batch = self.replay.sample_sequence() if self.vr_weight > 0.0 else None
            rp_batch = self.replay.sample_rp_sequence() if self.rp_weight > 0.0 else None
            return UnrealBatch(batch, pc_batch, rp_batch, vr_batch)

        def store_experience(self, observations, actions, rewards, terminals, values, action_log_prob, states):
            self.replay.insert(self.rollout_storage.observations, actions, rewards, terminals)
            super().store_experience(observations, actions, rewards, terminals, values, action_log_prob, states)

        def collect_experience(self):
            if not self.replay.full:
                while not self.replay.full:
                    result = super().collect_experience()

                logging.info('Experience replay full')
            else:
                result = super().collect_experience()
            return result

        def setup(self, stage):
            super().setup(stage)
            if stage == 'fit':
                self.replay = BatchExperienceReplay(self.num_agents, self.replay_size, self.num_steps)

        def get_input_for_pixel_control(self, x):
            return x['observation']

        def _loss_pixel_control(self, model, batch, device):
            observations, actions, rewards, terminals = batch
            masks = torch.ones(rewards.size(), dtype=torch.float32, device=device)
            initial_states = to_device(self._get_initial_states(masks.size()[0]), device)
            predictions, _ = model.pixel_control(observations, masks, initial_states)
            predictions[:, -1].mul_(1.0 - terminals[:, -2].float().view(*terminals[:, -2].size(), 1, 1, 1))
            pure_observations = self.get_input_for_pixel_control(observations)
            return pixel_control_loss(pure_observations, actions[:, :-1], predictions, self.pc_gamma, cell_size=model.pc_cell_size)

        def _loss_value_replay(self, model, batch, device):
            observations, actions, rewards, terminals = batch
            masks = torch.ones(rewards.size(), dtype=torch.float32, device=device)
            initial_states = to_device(self._get_initial_states(masks.size()[0]), device)
            predictions, _ = model.value_prediction(observations, masks, initial_states)
            predictions = predictions.squeeze(-1)
            predictions[:, -1].mul_(1.0 - terminals[:, -2].float())
            return value_loss(predictions, rewards[:, :-1], self.gamma)

        def _loss_reward_prediction(self, model, batch):
            observations, actions, rewards, terminals = batch
            predictions = model.reward_prediction(without_last_item(observations))
            return reward_prediction_loss(predictions, rewards[:, -1])

    return Unreal


Unreal = build_unreal(PAAC)
