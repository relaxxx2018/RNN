# !/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
import preprocess
from train_lstm import RNNModel
import random
from tqdm import tqdm
import os
import heapq
from cfg.option import Options
import random


# 用于预测
def predict(prefix, topK, model, device, category2id):
    state = None
    acc, mrr = 0, 0
    output = [category2id[prefix[0]]]
    for t in range(0, len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(category2id[prefix[t + 1]])
            list_Y = Y[0].cpu().detach().numpy().tolist()

            order = heapq.nlargest(50, range(len(list_Y)), list_Y.__getitem__)
            if category2id[prefix[t + 1]] in order[:topK]:
                acc += 1
            if category2id[prefix[t + 1]] in order:
                rank = order.index(category2id[prefix[t + 1]])
                mrr += 1 / (rank + 1)
            else:
                mrr += 0.01  # 为了评测更快一些只排序了前50个元素

    return acc / (len(prefix) - 1), mrr / (len(prefix) - 1)


if __name__ == '__main__':
    config_file = './cfg/example.cfg'
    params = Options(config_file)

    test_dataset_path = 'E:\\Bins\\DataSet\\UbiComp2016\\' + params.city.upper() \
                        + 'Data\\Checkins-Sequence-filter-leq5-spilt82-test-' + params.city.upper() + '.csv'
    dataset_path = 'E:\\Bins\\DataSet\\UbiComp2016\\' + params.city.upper() \
                   + 'Data\\Checkins-Sequence-filter-leq5-' + params.city.upper() + '.csv'
    pretrained_embedding_path = './output/sc@100_tky.txt'
    model_path = './output/sc@100_tky.pth'
    train_sequence, category2id = preprocess.load_data(dataset_path)
    pretrained_embedding = preprocess.load_pretrained_embedding(pretrained_embedding_path)
    pretrained_embedding_size = len(pretrained_embedding[0])

    test_sequence = preprocess.load_test_data(test_dataset_path)
    num_category = len(category2id.keys())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_size = 100
    lstm_layer = nn.LSTM(input_size=pretrained_embedding_size, hidden_size=embedding_size)
    model = RNNModel(lstm_layer, pretrained_embedding, num_category).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    main_path = 'result'
    topK = 5
    predict_acc, predict_mrr = 0, 0

    if not os.path.exists(main_path):
        os.makedirs(main_path)

    print(model_path)
    random.shuffle(test_sequence)
    for sequence in tqdm(test_sequence):
        acc, mrr = predict(sequence, topK, model, device, category2id)
        predict_acc += acc
        predict_mrr += mrr
    print(predict_acc / len(test_sequence))
    print(predict_mrr / len(test_sequence))
