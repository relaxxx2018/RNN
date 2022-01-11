# !/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
import utils
import math
from tqdm import tqdm
import time
import numpy as np


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, pretrained_embedding, num_category):
        super(RNNModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        # self.pretrained_embedding_size = len(pretrained_embedding[0])
        self.num_category = num_category
        # self.dense = nn.Linear(self.hidden_size, num_category)
        self.fc_hidden_size = rnn_layer.hidden_size
        self.out_linear = nn.Sequential(nn.Tanh(), nn.Linear(self.hidden_size, self.fc_hidden_size),
                                        nn.Tanh(), nn.Linear(self.fc_hidden_size, num_category))
        self.pretrained_embedding = torch.tensor(pretrained_embedding).to(self.device)
        self.state = None

    def forward(self, inputs, state):  # inputs: (batch, seq_len)
        # onehot向量表示
        # X = utils.encode_to_onehot(inputs, self.positions_number)
        X = utils.encode_to_embedding(inputs, self.pretrained_embedding)

        Y, self.state = self.rnn(torch.stack(X), state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size,num_hiddens)，它的输出形状为(num_steps * batch_size, vocab_size)
        output = self.out_linear(Y.view(-1, Y.shape[-1]))
        return output, self.state


def train(model, device, user_trajectory, num_epochs, num_steps, lr, clipping_theta, batch_size):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n = 0.0, 0
        data_batches = utils.data_iter_consecutive(user_trajectory, batch_size, num_steps, device)  # 相邻采样
        for X, Y in tqdm(data_batches, desc='epoch ' + str(epoch + 1)):
            if state is not None:
                # 使⽤用detach函数从计算图分离隐藏状态, 为了使模型参数的梯度计算只依赖一次迭代读取的小批量序列
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            (output, state) = model(X, state)
            # output: 形状为(num_steps * batch_size, vocab_size)
            # Y的形状是(batch_size, num_steps)，转置后再变成⻓长度为batch * num_steps 的向量，这样跟输出的⾏行⼀⼀对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())
            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            utils.grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        print('epoch %d, perplexity %f' % (epoch + 1, perplexity))
        time.sleep(0.1)


def lstm_trainer(category_sequence, category2id, pretrained_embedding, params, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_category = len(category2id.keys())
    pretrained_embedding_size = len(pretrained_embedding[0])

    print('hidden size:' + str(params.hidden_size))
    # 还可以在后面加层数和激活函数作为参数

    print('num_steps:' + str(params.num_steps))
    lstm_layer = nn.LSTM(input_size=pretrained_embedding_size, hidden_size=params.hidden_size)
    model = RNNModel(lstm_layer, pretrained_embedding, num_category).to(device)
    train(model, device, category_sequence,
          params.num_epochs, params.num_steps, params.lr, params.clipping_theta, params.batch_size)

    # save_path = params.embedding_output_path + '/lstm_s@' + str(embedding_size) + '_' + params.city + '.txt'
    # save(model.state_dict()['dense.Lweight'], category2id, save_path)
    # save_path = params.embedding_output_path + '/lstm_model@' + str(embedding_size) + '_' + params.city + '.pth'
    save_path = params.embedding_output_path + '/' + model_name + '@' + \
                str(params.init_embed_size) + '_' + params.city + '.pth'
    torch.save(model.state_dict(), save_path)


def save(vector_matrix, category2id, save_path):
    center_category_embedding = np.array(vector_matrix.cpu())
    f = open(save_path, 'w', encoding='utf-8')
    for i, category in enumerate(category2id.keys()):
        f.write(category + ',')
        f.write(','.join([str(_) for _ in center_category_embedding[i]]) + '\n')
    f.close()
