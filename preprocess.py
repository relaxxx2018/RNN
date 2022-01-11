# !/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from tqdm import tqdm
import random


def get_category_used(categories_used_path) -> (list, dict):
    with open(categories_used_path, 'r', encoding='utf-8') as f:
        categories = f.readlines()
    for i in range(len(categories)):
        categories[i] = categories[i].strip()
    category_index = range(len(categories))
    category2id = dict(zip(categories, category_index))

    return categories, category2id


# def load_data(path, categories_used_path):
#     print('loading train data .......')
#     category_sequence = []
#
#     with open(path, 'r', encoding='UTF-8') as train_f:
#         lines = train_f.readlines()
#     for i, line in enumerate(tqdm(lines, desc='loading')):
#         tokens = line.strip().split(',')
#         records = tokens[1:]
#         category_records = []
#         for record in records:
#             category = record[record.index('#') + 1: record.index('@')]
#             category_records.append(category)
#         category_sequence.extend(category_records)
#
#     categories, category2id = get_category_used(categories_used_path)
#     category_sequence = [category2id[i] for i in category_sequence]
#     print('There are', len(categories), 'categories, and', len(category_sequence), 'records')
#
#     return category_sequence, len(categories)


def load_data(path):
    """
    :param path: check-ins sequence
    :return:
        category: [[category_id, category_id, ...], [], ...]
        category2id: {category: category_id, ...}
    """
    print('loading data .......')
    category_sequence = []
    category_set = set()

    f = open(path, 'r', encoding='UTF-8')
    content = f.readline()
    while content != '':
        items = content.strip().split(',')
        for item in items[1:]:
            category = item[item.index('#') + 1: item.index('@')]
            category_set.add(category)
        content = f.readline()
    f.close()
    category2id = dict(zip(sorted(list(category_set)), range(len(category_set))))

    f = open(path, 'r', encoding='UTF-8')
    content = f.readline()
    while content != '':
        items = content.strip().split(',')
        temp1 = []
        for item in items[1:]:
            category = item[item.index('#') + 1: item.index('@')]
            temp1.append(category2id[category])
        category_sequence.extend(temp1)
        content = f.readline()
    f.close()

    return category_sequence, category2id


def load_test_data(path):
    category_sequence = []
    f = open(path, 'r', encoding='UTF-8')
    content = f.readline()
    while content != '':
        items = content.strip().split(',')
        temp1 = []
        for item in items[1:]:
            category = item[item.index('#') + 1: item.index('@')]
            temp1.append(category)
        category_sequence.append(temp1)
        content = f.readline()
    f.close()

    return category_sequence



def load_pretrained_embedding(pretrained_embedding_path):
    category_embedding = []
    with open(pretrained_embedding_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        temp = line.strip().split(',')
        embedding = [float(temp[i]) for i in range(1, len(temp))]
        category_embedding.append(embedding)
    return category_embedding


def one_hot(x, n_class):
    # X shape: (batch), output shape: (batch, positions_number)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=torch.float32, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


def encode_to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch,n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


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




