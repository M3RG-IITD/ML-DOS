import sys
import os
import torch
import pandas as pd
import argparse
import pyhocon
import random

from src.dataCenter import *
from src.utils import *
from src.models import *

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')
parser.add_argument('--dataSet', type=str, default='cora')
parser.add_argument('--Val', type=int, default=10)
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--b_sz', type=int, default=250)
parser.add_argument('--seed', type=int, default=824)
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--learn_method', type=str, default='sup')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--config', type=str, default='./src/experiments.conf')
args = parser.parse_args()

if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)

if __name__ == '__main__':
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	# load config file
	config = pyhocon.ConfigFactory.parse_file(args.config)
	# load data
	ds = args.dataSet
	ds_val=args.Val
	dataCenter = DataCenter(config)
	dataCenter.load_dataSet(ds)
	features = torch.FloatTensor(getattr(dataCenter, ds+'_feats')[0]).to(device)

	dataCenter_val = DataCenter(config)
	dataCenter_val.load_dataSet(ds+"_val")
	features_val = torch.FloatTensor(getattr(dataCenter_val, ds+"_val"+'_feats')[0]).to(device)
	######

	graphSage = GraphSage(config['setting.num_layers'], features.size(1), config['setting.hidden_emb_size'], device, gcn=False, agg_func=args.agg_func)
	graphSage.to(device)

	num_labels = len(set(getattr(dataCenter, ds+'_labels')[0]))
	classification = Classification(config['setting.hidden_emb_size'], num_labels)
	classification.to(device)

	unsupervised_loss = [UnsupervisedLoss(getattr(dataCenter, ds+'_adj_lists')[i], getattr(dataCenter, ds+'_train')[i], device) for i in range(0,config['setting.N_graph_train_index_end']-config['setting.N_graph_train_index_start'],1)]
	unsupervised_loss_val = [UnsupervisedLoss(getattr(dataCenter_val, ds+"_val"+'_adj_lists')[i], getattr(dataCenter_val, ds+"_val"+'_train')[i], device) for i in range(0,config['setting.N_graph_val_index_end']-config['setting.N_graph_val_index_start'],1)]

	if args.learn_method == 'sup':
		print('GraphSage with Supervised Learning')
	elif args.learn_method == 'plus_unsup':
		print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
	else:
		print('GraphSage with Net Unsupervised Learning')
	Best_loss=100000000
	for epoch in range(args.epochs):
		if(epoch==0):
			for Gid in range(0,args.Val):
				print("Evaluation_Metric for graph:"+str(config['setting.N_graph_val_index_start']+Gid))
				nodes = getattr(dataCenter_val, ds+"_val"+'_train')[Gid]
				features = torch.FloatTensor(getattr(dataCenter_val, ds+"_val"+'_feats')[Gid]).to(device)
				adj_list=getattr(dataCenter_val, ds+"_val"+'_adj_lists')[Gid]
				nodes_batch = np.asarray(list(unsupervised_loss_val[Gid].extend_nodes(nodes, num_neg=100)))
				Final_embs=graphSage(np.asarray(nodes_batch),features,adj_list)
				unsupervised_loss_val[Gid].get_metric_sage(Final_embs,nodes_batch)	
		print('----------------------EPOCH %d-----------------------' % epoch)
		graphSage, classification = apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, args.b_sz, args.unsup_loss, device, args.learn_method)
		
		if(epoch%5==0):
			torch.save(graphSage,'models/Final_model_'+str(epoch)+'.torch')
			print("--------------Validating---------------------")
			graphSage, classification,Loss = Val_model(dataCenter_val, ds+"_val", graphSage, classification, unsupervised_loss_val, args.b_sz, args.unsup_loss, device, args.learn_method)
			
			if(Best_loss>Loss):
				Best_loss=Loss
				for Gid in range(0,args.Val):
					nodes = getattr(dataCenter_val, ds+"_val"+'_train')[Gid]
					features = torch.FloatTensor(getattr(dataCenter_val, ds+"_val"+'_feats')[Gid]).to(device)
					adj_list=getattr(dataCenter_val, ds+"_val"+'_adj_lists')[Gid]
					nodes_batch = np.asarray(list(unsupervised_loss_val[Gid].extend_nodes(nodes, num_neg=100)))
					Final_embs=graphSage(np.asarray(nodes_batch),features,adj_list)
					node_scores=unsupervised_loss_val[Gid].get_metric_sage(Final_embs,nodes_batch)
					print("Evaluation_Metric for graph :"+str(config['setting.N_graph_val_index_start']+Gid))
					Extra_node_data=getattr(dataCenter_val, ds+"_val"+'_Ext')[Gid]
					t_np = Final_embs.detach().cpu().numpy() 
					df = pd.DataFrame(t_np) 
					df1 = pd.DataFrame(Extra_node_data) 
					df2=pd.DataFrame(node_scores)
					df.to_csv("Fin_embs_"+str(config['setting.N_graph_val_index_start']+Gid)+".csv",index=False) #save to file
					df1.to_csv("Fin_Ext_"+str(config['setting.N_graph_val_index_start']+Gid)+".csv",index=False) #save to file
					df2.to_csv("Node_Scores_"+str(config['setting.N_graph_val_index_start']+Gid)+".csv",index=False) #save to file
	
					
