from ..a2c.a2c import A2CTrainer
from ..optim.kfac import KFACOptimizer

class ACKTRTrainer(A2CTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._optimizer = None

    def _build_train(self, model, main_device):
        optimizer = KFACOptimizer(model, self.learning_rate)
        @pytorch_call(main_device)
        def train(observations, returns, actions, masks, states = []):
            # Update optimizer's learning rate
            optimizer.lr = self.learning_rate

            policy_logits, value, _ = model(observations, masks, states)

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
            optimizer.step()

            return loss.item(), action_loss.item(), value_loss.item(), dist_entropy.item()

        return train
