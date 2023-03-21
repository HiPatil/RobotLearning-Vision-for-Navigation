import random
import torch


def select_greedy_action(state, policy_net, action_size):
    """ Select the greedy action
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    Returns
    -------
    int
        ID of selected action
    """

    # TODO: Select greedy action
    state = torch.FloatTensor(state).permute(0, 3, 1, 2).cuda()
    q_values = policy_net(state)
    return torch.argmax(q_values).item()

def select_exploratory_action(state, policy_net, action_size, exploration, t):
    """ Select an action according to an epsilon-greedy exploration strategy
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    exploration: LinearSchedule
        linear exploration schedule
    t: int
        current time-step
    Returns
    -------
    int
        ID of selected action
    """

    # TODO: Select exploratory action
    if random.random() < exploration.value(t):
        return random.randrange(action_size)
    else:
        state = torch.FloatTensor(state).permute(0, 3, 1, 2).cuda()
        q_values = policy_net(state)
        return torch.argmax(q_values).item()

def get_action_set():
    """ Get the list of available actions
    Returns
    -------
    list
        list of available actions
    """
    return [[-1.0, 0.05, 0], [1.0, 0.05, 0], [0, 0.5, 0], [0, 0, 1.0]]
