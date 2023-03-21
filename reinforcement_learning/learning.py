import numpy as np
import torch
import torch.nn.functional as F


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step
    """ Steps: 
        1. Sample transitions from replay_buffer
        2. Compute Q(s_t, a)
        3. Compute \max_a Q(s_{t+1}, a) for all next states.
        4. Mask next state values where episodes have terminated
        5. Compute the target
        6. Compute the loss
        7. Calculate the gradients
        8. Clip the gradients
        9. Optimize the model
    """
    clipping_value = 3

    state, action, reward, next_state, done= replay_buffer.sample(batch_size)
    state = torch.FloatTensor(state).to(device).permute(0, 3, 1, 2)
    next_state = torch.FloatTensor(next_state).to(device).permute(0, 3, 1, 2)
    action = torch.tensor(action).to(device)
    reward=torch.tensor(reward).to(device)
    done=torch.FloatTensor(done).to(device)



    policy_q_values = policy_net(state)
    target_q_values = target_net(next_state).detach()

    policy_q = policy_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    target_q = target_q_values.max(1)[0]

    expected_q = reward + gamma * target_q * (1 - done)

    loss = F.mse_loss(policy_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), clipping_value)
    optimizer.step()


    return loss.item()


def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    # TODO: Update target network
    target_net.load_state_dict(policy_net.state_dict())
