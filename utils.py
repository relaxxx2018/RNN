# !/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from tqdm import tqdm


def one_hot(x, n_class):
    # X shape: (batch), output shape: (batch, positions_number)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=torch.float32, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


def encode_to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch,n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


def encode_to_embedding(X, pretrained_embedding):
    temp = []
    for i in range(X.shape[1]):
        x = X[:, i].long()
        res = pretrained_embedding[x]
        temp.append(res)
    return temp


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


def data_iter_consecutive(user_trajectory, batch_size, num_steps, device):
    data_len = len(user_trajectory)
    user_trajectory = torch.tensor(user_trajectory, dtype=torch.float32, device=device)
    batch_len = data_len // batch_size
    indices = user_trajectory[0:batch_size * batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    data_batch = []
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        # yield X, Y  # 迭代器
        batch = [X, Y]
        data_batch.append(batch)
    return data_batch
