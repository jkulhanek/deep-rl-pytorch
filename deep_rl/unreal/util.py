import torch
import torch.nn.functional as F


def autocrop_observations(observations, cell_size):
    shape = (observations[3], observations[4])
    new_shape = tuple(map(lambda x: (x // cell_size) * cell_size, shape))
    margin3_top = (shape[0] - new_shape[0]) // 2
    margin3_bottom = shape[0] - new_shape[0] - margin3_top
    margin4_top = (shape[1] - new_shape[1]) // 2
    margin4_bottom = shape[1] - new_shape[1] - margin4_top
    return observations[:,:,:,margin3_top:-margin3_bottom,margin4_top:-margin4_bottom]

def pixel_control_reward(observations, cell_size = 4):
    '''
    Args:
    observations: A tensor of shape `[B,T+1,C,H,W]`, where
      * `T` is the sequence length, `B` is the batch size.
      * `H` is height, `W` is width.
      * `C...` is at least one channel dimension (e.g., colour, stack).
      * `T` and `B` can be statically unknown.
    cell_size: The size of each cell.
    Returns:
        shape (B, T, 1, H / cell_size, W / cell_size)
    '''
    with torch.no_grad():
        observations = autocrop_observations(observations, cell_size)
        abs_observation_diff = observations[1:] - observations[:-1]
        abs_observation_diff.abs_()
        obs_shape = abs_observation_diff.size()
        abs_diff = abs_observation_diff.view(-1, *obs_shape[2:])
        
        avg_abs_diff = F.avg_pool1d(abs_diff, cell_size, stride=cell_size)
        avg_abs_diff = avg_abs_diff.mean(1, keepdim = True)
        return avg_abs_diff.view(*obs_shape[:2] + avg_abs_diff.size()[1:])



def pixel_control_loss(observations, actions, action_values, gamma = 0.9, cell_size = 4):
    action_value_shape = action_values.size()
    batch_shape = actions.size()[:2]
    with torch.no_grad():
        T = observations.size()[1] - 1
        pseudo_rewards = pixel_control_reward(observations, cell_size)
        last_rewards = action_values.max(-1)
        for i in reversed(range(T)):
            previous_rewards = last_rewards if i == T else pseudo_rewards[:, i + 1]
            pseudo_rewards[:, i].add_(gamma * previous_rewards)

    q_actions = actions.view(*batch_shape + [1, 1, 1, 1]).repeat(1, 1, 1, action_value_shape[3], action_value_shape[4], 1)
    q_actions = torch.gather(action_values, -1, q_actions).squeeze(-1)

    loss = F.mse_loss(pseudo_rewards, q_actions)
    return loss

def reward_prediction_loss(predictions, rewards):
    with torch.no_grad():
        target = torch.stack((rewards == 0, rewards > 0, rewards < 0), dim = 2)
    return F.binary_cross_entropy_with_logits(predictions, target)
