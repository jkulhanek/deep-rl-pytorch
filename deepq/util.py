import torch
def qlearning(q, actions, rewards, pcontinues, q_next_online_net):
    with torch.no_grad():
        target = rewards + pcontinues * torch.max(q_next_online_net, dim = -1)[0]

    actions = actions.long()
    q_actions = torch.gather(q, -1, actions.unsqueeze(-1)).squeeze(-1)

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target - q_actions
    loss = 0.5 * (td_error ** 2)
    loss = loss.mean()
    return loss

def double_qlearning(q, actions, rewards, pcontinues, q_next, q_next_selector):
    with torch.no_grad():
        indices = torch.argmax(q_next_selector, dim = -1, keepdim = True)
        target = rewards + pcontinues * torch.gather(q_next, -1, indices).squeeze(-1)

    actions = actions.long()
    q_actions = torch.gather(q, -1, actions.unsqueeze(-1)).squeeze(-1)

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target - q_actions
    loss = 0.5 * (td_error ** 2)
    loss = loss.mean()
    return loss