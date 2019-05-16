from itertools import accumulate as accum

import matplotlib.pyplot as plt
import numpy as np
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)


def accumulate(x, discount_rate=1):
    reversed_x = x[::-1]
    accumulated_x = np.asarray(list(accum(reversed_x, lambda prev, curr: prev * discount_rate + curr)))[::-1]
    return accumulated_x


def batch_normalize(x):
    means = np.mean(x, axis=1)
    stds = np.std(x, axis=1) + 1e-10
    return (x - means[:, np.newaxis]) / stds[:, np.newaxis]


def to_tensor_long(numpy_array):
    return torch.LongTensor(numpy_array).to(DEVICE)


def to_tensor(numpy_array):
    return torch.Tensor(numpy_array).to(DEVICE)


def run_model_with_no_grad(model, inputs, to_numpy=False, eval_mode=True):
    is_train = model.training
    model = model.eval() if eval_mode else model.train()

    with torch.no_grad():
        outputs = model(inputs)

    model.train(mode=is_train)
    if to_numpy:
        outputs = outputs.cpu().numpy()
    return outputs


def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def transpose_list(mylist):
    return list(map(list, zip(*mylist)))


def to_full(tensors):
    if type(tensors) is torch.Tensor:
        batch_size = tensors.size(0)
        return tensors.view(batch_size, -1)

    return torch.cat(tensors, dim=1)


def print_ratio_for_debugging(i_episode, new_log_probs, sampled_old_log_probs):
    ratio = torch.exp(new_log_probs.detach() - sampled_old_log_probs.detach())
    if i_episode % 10:
        print(ratio.mean())


def plotter(env_name, num_episodes, rewards_list, ylim):
    '''
    Used to plot the average over time
    :param env_name:
    :param num_episodes:
    :param rewards_list:
    :param ylim:
    :return:
    '''
    x = np.arange(0, num_episodes)
    y = np.asarray(rewards_list)
    plt.plot(x, y)
    plt.ylim(top=ylim + 3)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Avg Rewards Last 100 Episodes")
    plt.title("Rewards Over Time For %s" % env_name)
    plt.savefig("progress.png")
    plt.close()


def raw_score_plotter(scores):
    '''
    used to plot the raw score
    :param scores:
    :return:
    '''
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Episode Rewards')
    plt.xlabel('Number of Episodes')
    plt.title("Raw Scores Over Time")
    plt.savefig("RawScore.png")
    plt.close()
