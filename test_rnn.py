# !/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
import preprocess
from train_rnn import RNNModel
import random
from tqdm import tqdm
import os
import heapq


# 用于预测
def predict(prefix, num_predict, model, device, position2id, id2position):
	state = None
	output = [position2id[prefix[0]]]
	for t in range(num_predict + len(prefix) - 1):
		X = torch.tensor([output[-1]], device=device).view(1, 1)
		if state is not None:
			if isinstance(state, tuple):
				state = (state[0].to(device), state[1].to(device))
			else:
				state = state.to(device)

		(Y, state) = model(X, state)
		if t < len(prefix) - 1:
			output.append(position2id[prefix[t + 1]])
		else:
			output.append(int(Y.argmax(dim=1).item()))
			list_Y = Y[0].cpu().detach().numpy().tolist()
			# 获取下标
			indexes = heapq.nlargest(5, range(len(list_Y)), list_Y.__getitem__)

	return [id2position[i] for i in indexes]


if __name__ == '__main__':

	layers = 0
	test_dataset_path = '../data/LinyiTrajectoryTest.csv'
	train_dataset_path = '../data/LinyiTrajectoryTrain.csv'
	position2id, id2position, positions_number = preprocess.load_data(train_dataset_path)[1:]

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	num_hiddens = 128

	rnn_layer = nn.RNN(input_size=positions_number, hidden_size=num_hiddens)
	model = RNNModel(rnn_layer, positions_number).to(device)
	model.load_state_dict(torch.load('model/rnn_model_'+str(layers)+'.pth', map_location=device))
	with open(test_dataset_path, 'r', encoding='UTF-8') as f:
		lines = f.readlines()
	test_trajectory_list = []
	for i, line in enumerate(lines):
		tokens = line.strip().split(',')
		records = tokens[1:-1]
		records_without_time = []
		for record in records:
			records_without_time.append(record[:record.index('@')])
		test_trajectory_list.append(records_without_time)

	main_path = 'result'
	if not os.path.exists(main_path):
		os.makedirs(main_path)
	with open(main_path + '/rnn_result_'+str(layers)+'.txt', 'a') as f:
		for test_trajectory in tqdm(test_trajectory_list):
			try:
				pre = test_trajectory
				res = predict(pre, 1, model, device, position2id, id2position)
				for location in res[:-1]:
					f.write(location)
					f.write('#')
				f.write(res[-1])
				f.write('\n')
			except:
				f.write('empty\n')

