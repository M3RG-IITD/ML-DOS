import sys
import os
import torch
import random
import math

from sklearn.utils import shuffle
from sklearn.metrics import f1_score

import torch.nn as nn
import numpy as np

def evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, cur_epoch):
	test_nodes = getattr(dataCenter, ds+'_test')
	val_nodes = getattr(dataCenter, ds+'_val')
	labels = getattr(dataCenter, ds+'_labels')

	models = [graphSage, classification]

	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				param.requires_grad = False
				params.append(param)

	embs = graphSage(val_nodes)
	logists = classification(embs)
	_, predicts = torch.max(logists, 1)
	labels_val = labels[val_nodes]
	assert len(labels_val) == len(predicts)
	comps = zip(labels_val, predicts.data)

	vali_f1 = f1_score(labels_val, predicts.cpu().data, average="micro")
	print("Validation F1:", vali_f1)

	if vali_f1 > max_vali_f1:
		max_vali_f1 = vali_f1
		embs = graphSage(test_nodes)
		logists = classification(embs)
		_, predicts = torch.max(logists, 1)
		labels_test = labels[test_nodes]
		assert len(labels_test) == len(predicts)
		comps = zip(labels_test, predicts.data)

		test_f1 = f1_score(labels_test, predicts.cpu().data, average="micro")
		print("Test F1:", test_f1)

		for param in params:
			param.requires_grad = True

		torch.save(models, 'models/model_best_{}_ep{}_{:.4f}.torch'.format(name, cur_epoch, test_f1))

	for param in params:
		param.requires_grad = True

	return max_vali_f1

def get_gnn_embeddings(gnn_model, dataCenter, ds):
    print('Loading embeddings from trained GraphSAGE model.')
    features = np.zeros((len(getattr(dataCenter, ds+'_labels')), gnn_model.out_size))
    nodes = np.arange(len(getattr(dataCenter, ds+'_labels'))).tolist()
    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
        embs_batch = gnn_model(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    print('Embeddings loaded.')
    return embs.detach()

def train_classification(dataCenter, graphSage, classification, ds, device, max_vali_f1, name, epochs=800):
	print('Training Classification ...')
	c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)
	# train classification, detached from the current graph
	#classification.init_params()
	b_sz = 50
	train_nodes = getattr(dataCenter, ds+'_train')
	labels = getattr(dataCenter, ds+'_labels')
	features = get_gnn_embeddings(graphSage, dataCenter, ds)
	for epoch in range(epochs):
		train_nodes = shuffle(train_nodes)
		batches = math.ceil(len(train_nodes) / b_sz)
		visited_nodes = set()
		for index in range(batches):
			nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
			visited_nodes |= set(nodes_batch)
			labels_batch = labels[nodes_batch]
			embs_batch = features[nodes_batch]

			logists = classification(embs_batch)
			loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss /= len(nodes_batch)
			# print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))

			loss.backward()
			
			nn.utils.clip_grad_norm_(classification.parameters(), 5)
			c_optimizer.step()
			c_optimizer.zero_grad()

		max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
	return classification, max_vali_f1

def apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, b_sz, unsup_loss, device, learn_method):
	#DataCenter has a list of all required variables
	#unsupervised_loss is a list of functions
	if unsup_loss == 'margin':
		num_neg = 50
	elif unsup_loss == 'normal':
		num_neg = 50
	else:
		print("unsup_loss can be only 'margin' or 'normal'.")
		sys.exit(1)
	
	models = [graphSage, classification]
	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				params.append(param)

	optimizer = torch.optim.Adam(params, lr=0.01)
	optimizer.zero_grad()
	for model in models:
		model.zero_grad()

	N=len(getattr(dataCenter, ds+'_train'))
	graph_data_indexes=shuffle([i for i in range(N)])
	graph_batch_len=5									#Batch_size for multiple graphs
	counter=0
	for k in graph_data_indexes:
		if(counter%graph_batch_len==0):
			Batchloss=0 		#after every n_bacth_len steps
		
		test_nodes = getattr(dataCenter, ds+'_test')[k]
		val_nodes = getattr(dataCenter, ds+'_val')[k]
		train_nodes = getattr(dataCenter, ds+'_train')[k]
		labels = getattr(dataCenter, ds+'_labels')[k]

		
		train_nodes = shuffle(train_nodes)
		
		batches = math.ceil(len(train_nodes) / b_sz)

		visited_nodes = set()
		for index in range(batches):
			nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]

			# extend nodes batch for unspervised learning
			# no conflicts with supervised learning
			nodes_batch = np.asarray(list(unsupervised_loss[k].extend_nodes(nodes_batch, num_neg=num_neg)))
			visited_nodes |= set(nodes_batch)

			# get ground-truth for the nodes batch
			labels_batch = labels[nodes_batch]

			# feed nodes batch to the graphSAGE
			# returning the nodes embeddings
			embs_batch = graphSage(nodes_batch,torch.FloatTensor(getattr(dataCenter, ds+'_feats')[k]).to(device), getattr(dataCenter, ds+'_adj_lists')[k])

			if learn_method == 'sup':
				# superivsed learning
				logists = classification(embs_batch)
				loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
				loss_sup /= len(nodes_batch)
				loss = loss_sup
			elif learn_method == 'plus_unsup':
				# superivsed learning
				logists = classification(embs_batch)
				loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
				loss_sup /= len(nodes_batch)
				# unsuperivsed learning
				if unsup_loss == 'margin':
					loss_net = unsupervised_loss[k].get_loss_margin(embs_batch, nodes_batch)
				elif unsup_loss == 'normal':
					loss_net = unsupervised_loss[k].get_loss_sage(embs_batch, nodes_batch)
				loss = loss_sup + loss_net
			else:
				if unsup_loss == 'margin':
					loss_net = unsupervised_loss[k].get_loss_margin(embs_batch, nodes_batch)
				elif unsup_loss == 'normal':
					loss_net = unsupervised_loss[k].get_loss_sage(embs_batch, nodes_batch)
				loss = loss_net
			
			Batchloss+=loss #Currently adds loss of eatch node_batch, can be taken to outer loop, to take only one loss per graph 
		if((counter+1)%graph_batch_len==0):
			Batchloss/=graph_batch_len		#Do after every n steps , divide loss by n
			print(counter+1,'Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, Batchloss.item(), len(visited_nodes), len(train_nodes)))			
			Batchloss.backward() #after every n steps
			for model in models:
				nn.utils.clip_grad_norm_(model.parameters(), 5)
			optimizer.step()	
			#Keep a set of graph not in train
			optimizer.zero_grad()
			for model in models:
				model.zero_grad()
		counter+=1

	return graphSage, classification

def Val_model(dataCenter, ds, graphSage, classification, unsupervised_loss, b_sz, unsup_loss, device, learn_method):
	#DataCenter has a list of all required variables
	#unsupervised_loss is a list of functions
	if unsup_loss == 'margin':
		num_neg = 6
	elif unsup_loss == 'normal':
		num_neg = 100
	else:
		print("unsup_loss can be only 'margin' or 'normal'.")
		sys.exit(1)
	
	models = [graphSage, classification]
	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				params.append(param)

	N=len(getattr(dataCenter, ds+'_train'))
	graph_data_indexes=shuffle([i for i in range(N)])
	graph_batch_len=5									#Batch_size for multiple graphs
	counter=0
	for k in graph_data_indexes:
		if(counter%graph_batch_len==0):
			Batchloss=0 		#after every n_bacth_len steps
		
		test_nodes = getattr(dataCenter, ds+'_test')[k]
		val_nodes = getattr(dataCenter, ds+'_val')[k]
		train_nodes = getattr(dataCenter, ds+'_train')[k]
		labels = getattr(dataCenter, ds+'_labels')[k]

		
		train_nodes = shuffle(train_nodes)
		
		batches = math.ceil(len(train_nodes) / b_sz)

		visited_nodes = set()
		for index in range(batches):
			nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]

			# extend nodes batch for unspervised learning
			# no conflicts with supervised learning
			nodes_batch = np.asarray(list(unsupervised_loss[k].extend_nodes(nodes_batch, num_neg=num_neg)))
			visited_nodes |= set(nodes_batch)

			# get ground-truth for the nodes batch
			labels_batch = labels[nodes_batch]

			# feed nodes batch to the graphSAGE
			# returning the nodes embeddings
			embs_batch = graphSage(nodes_batch,torch.FloatTensor(getattr(dataCenter, ds+'_feats')[k]).to(device), getattr(dataCenter, ds+'_adj_lists')[k])

			if learn_method == 'sup':
				# superivsed learning
				logists = classification(embs_batch)
				loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
				loss_sup /= len(nodes_batch)
				loss = loss_sup
			elif learn_method == 'plus_unsup':
				# superivsed learning
				logists = classification(embs_batch)
				loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
				loss_sup /= len(nodes_batch)
				# unsuperivsed learning
				if unsup_loss == 'margin':
					loss_net = unsupervised_loss[k].get_loss_margin(embs_batch, nodes_batch)
				elif unsup_loss == 'normal':
					loss_net = unsupervised_loss[k].get_loss_sage(embs_batch, nodes_batch)
				loss = loss_sup + loss_net
			else:
				if unsup_loss == 'margin':
					loss_net = unsupervised_loss[k].get_loss_margin(embs_batch, nodes_batch)
				elif unsup_loss == 'normal':
					loss_net = unsupervised_loss[k].get_loss_sage(embs_batch, nodes_batch)
				loss = loss_net
			
			Batchloss+=loss #Currently adds loss of eatch node_batch, can be taken to outer loop, to take only one loss per graph 
		
		
		if((counter+1)%graph_batch_len==0):
			Batchloss/=graph_batch_len		#Do after every n steps , divide loss by n
			print(counter+1,'Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, Batchloss.item(), len(visited_nodes), len(train_nodes)))			
			 #after every n steps	
		counter+=1

	return graphSage, classification, Batchloss

