import torch

from .a2c import A2C
from ..optim.kfac import KFACOptimizer
from ..common.pytorch import pytorch_call


class ACKTR(A2C):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_train(self, model, main_device):
        optimizer = KFACOptimizer(model, self.learning_rate)

        @pytorch_call(main_device)
        def train(observations, returns, actions, masks, states=[]):
            # Update optimizer's learning rate
            optimizer.lr = self.learning_rate

            policy_logits, value, _ = model(observations, masks, states)

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
            if optimizer.steps % optimizer.Ts == 0:
                # Sampled fisher, see Martens 2014
                model.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()
                value_noise = torch.randn(value.size(), device=value.device)
                sample_value = value + value_noise
                vf_fisher_loss = -(value - sample_value.detach()).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                optimizer.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                optimizer.acc_stats = False

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return loss.item(), action_loss.item(), value_loss.item(), dist_entropy.item()

        return train
