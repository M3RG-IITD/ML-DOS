import sys
import os

from collections import defaultdict
import numpy as np
import networkx as nx
import pandas as pd

class DataCenter(object):
	"""docstring for DataCenter"""
	def __init__(self, config):
		super(DataCenter, self).__init__()
		self.config = config

	def one_hot(self,x, class_count):
		return torch.eye(class_count)[x,:]
	""" x = [0,2,5,4]
		class_count = 8
		one_hot(x,class_count)
		tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
				[0., 0., 1., 0., 0., 0., 0., 0.],
				[0., 0., 0., 0., 0., 1., 0., 0.],
				[0., 0., 0., 0., 1., 0., 0., 0.]])
	"""

	def load_dataSet(self, dataSet='cora'):#Add graph_id
		if dataSet=='DualLJ':
			DualLJ = self.config['file_path.DualLJ']	#pass id
			test_indexs_i=[]
			val_indexs_i=[]
			train_indexs_i=[]
			feat_data_i=[]
			labels_i=[]
			adj_lists_i=[]
			for k in range(100):
				G :nx.Graph=nx.read_graphml(DualLJ+"/"+str(k)+"Dual.gml")
				Bin_type_label=[[1,0,0],[0,1,0],[1,0,0]]	#Type 2=[1,0,0] ,3=[0,1,0],4=[0,0,1]
				feat_data = []
				labels = [] # label sequence of node
				node_map = {} # map node to Node_ID
				label_map = {} # map label to Label_ID
				i=0
				for node in G:
					features=list(G.nodes[node].values())
					feat_data.append(Bin_type_label[features[0]-2]+[float(x) for x in features[1:]])
					node_map[node] = i
					if not features[0] in label_map:
						label_map[features[0]] = len(label_map)
					labels.append(label_map[features[0]])
					i+=1
				feat_data = np.asarray(feat_data)
				labels = np.asarray(labels, dtype=np.int64)
				adj_lists = defaultdict(set)
				for edge in G.edges:
					e1=node_map[edge[0]]
					e2=node_map[edge[1]]
					adj_lists[e1].add(e2)
					adj_lists[e2].add(e1)
				

				assert len(feat_data) == len(labels) == len(adj_lists)
				test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0],2*feat_data.shape[0],2*feat_data.shape[0])
				test_indexs_i+=[test_indexs]
				val_indexs_i+=[val_indexs]
				train_indexs_i+=[train_indexs]
				feat_data_i+=[feat_data]
				labels_i+=[labels]
				adj_lists_i+=[adj_lists]


			setattr(self, dataSet+'_test', test_indexs_i)
			setattr(self, dataSet+'_val', val_indexs_i)
			setattr(self, dataSet+'_train', train_indexs_i)

			setattr(self, dataSet+'_feats', feat_data_i)
			setattr(self, dataSet+'_labels', labels_i)
			setattr(self, dataSet+'_adj_lists', adj_lists_i)
		
		if dataSet=='NormLJ':
			NormLJ = self.config['file_path.NormLJ']
			test_indexs_i=[]
			val_indexs_i=[]
			train_indexs_i=[]
			feat_data_i=[]
			labels_i=[]
			adj_lists_i=[]
			Ext_data_i=[]
			for k in range(self.config['setting.N_graph_train_index_start'],self.config['setting.N_graph_train_index_end'],1):
				G :nx.Graph=nx.read_graphml(NormLJ+"/"+str(k)+"Norm.gml")
				Bin_type_label=[[1,0],[0,1]]
				feat_data = []
				Ext_data=[]
				labels = [] # label sequence of node
				node_map = {} # map node to Node_ID
				label_map = {} # map label to Label_ID
				i=0
				for node in G:
					features=list(G.nodes[node].values())
					feat_data.append(Bin_type_label[features[0]-1]+[float(x) for x in features[1:4]])
					Ext_data.append([float(x) for x in features])

					node_map[node] = i
					if not features[0] in label_map:
						label_map[features[0]] = len(label_map)
					labels.append(label_map[features[0]])
					i+=1
			
				feat_data = np.asarray(feat_data)
				Ext_data=np.asarray(Ext_data)
				labels = np.asarray(labels, dtype=np.int64)
				adj_lists = defaultdict(set)
				for edge in G.edges:
					e1=node_map[edge[0]]
					e2=node_map[edge[1]]
					adj_lists[e1].add(e2)
					adj_lists[e2].add(e1)
				
				assert len(feat_data) == len(labels) == len(adj_lists)
				test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0],2*feat_data.shape[0],2*feat_data.shape[0])
				test_indexs_i+=[test_indexs]
				val_indexs_i+=[val_indexs]
				train_indexs_i+=[train_indexs]
				feat_data_i+=[feat_data]
				Ext_data_i+=[Ext_data]
				labels_i+=[labels]
				adj_lists_i+=[adj_lists]
			
			setattr(self, dataSet+'_test', test_indexs_i)
			setattr(self, dataSet+'_val', val_indexs_i)
			setattr(self, dataSet+'_train', train_indexs_i)
			setattr(self, dataSet+'_feats', feat_data_i)
			setattr(self, dataSet+'_Ext', Ext_data_i)
			setattr(self, dataSet+'_labels', labels_i)
			setattr(self, dataSet+'_adj_lists', adj_lists_i)
		
		if dataSet=='NormLJ_val':
			NormLJ = self.config['file_path.NormLJ']
			test_indexs_i=[]
			val_indexs_i=[]
			train_indexs_i=[]
			feat_data_i=[]
			Ext_data_i=[]
			labels_i=[]
			adj_lists_i=[]
			for k in range(self.config['setting.N_graph_val_index_start'],self.config['setting.N_graph_val_index_end'],1):
				
				G :nx.Graph=nx.read_graphml(NormLJ+"/"+str(k)+"Norm.gml")
				Bin_type_label=[[1,0],[0,1]]
				feat_data = []
				Ext_data=[]
				labels = [] # label sequence of node
				node_map = {} # map node to Node_ID
				label_map = {} # map label to Label_ID
				i=0
				for node in G:
					features=list(G.nodes[node].values())
					feat_data.append(Bin_type_label[features[0]-1]+[float(x) for x in features[1:4]])
					Ext_data.append([float(x) for x in features])
					node_map[node] = i
					if not features[0] in label_map:
						label_map[features[0]] = len(label_map)
					labels.append(label_map[features[0]])
					i+=1
				
				feat_data = np.asarray(feat_data)
				Ext_data=np.asarray(Ext_data)
				labels = np.asarray(labels, dtype=np.int64)
				adj_lists = defaultdict(set)
				for edge in G.edges:
					e1=node_map[edge[0]]
					e2=node_map[edge[1]]
					adj_lists[e1].add(e2)
					adj_lists[e2].add(e1)
					

				assert len(feat_data) == len(labels) == len(adj_lists)
				test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0],2*feat_data.shape[0],2*feat_data.shape[0])
				test_indexs_i+=[test_indexs]
				val_indexs_i+=[val_indexs]
				train_indexs_i+=[train_indexs]
				feat_data_i+=[feat_data]
				Ext_data_i+=[Ext_data]
				labels_i+=[labels]
				adj_lists_i+=[adj_lists]

			setattr(self, dataSet+'_test', test_indexs_i)
			setattr(self, dataSet+'_val', val_indexs_i)
			setattr(self, dataSet+'_train', train_indexs_i)
			setattr(self, dataSet+'_feats', feat_data_i)
			setattr(self, dataSet+'_Ext', Ext_data_i)
			setattr(self, dataSet+'_labels', labels_i)
			setattr(self, dataSet+'_adj_lists', adj_lists_i)
			
		if dataSet == 'cora':
			cora_content_file = self.config['file_path.cora_content']
			cora_cite_file = self.config['file_path.cora_cite']

			feat_data = []
			labels = [] # label sequence of node
			node_map = {} # map node to Node_ID
			label_map = {} # map label to Label_ID
			with open(cora_content_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					feat_data.append([float(x) for x in info[1:-1]])
					node_map[info[0]] = i
					if not info[-1] in label_map:
						label_map[info[-1]] = len(label_map)
					labels.append(label_map[info[-1]])
			feat_data = np.asarray(feat_data)
			labels = np.asarray(labels, dtype=np.int64)
			
			adj_lists = defaultdict(set)
			with open(cora_cite_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					assert len(info) == 2
					paper1 = node_map[info[0]]
					paper2 = node_map[info[1]]
					adj_lists[paper1].add(paper2)
					adj_lists[paper2].add(paper1)

			assert len(feat_data) == len(labels) == len(adj_lists)
			test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

			setattr(self, dataSet+'_test', test_indexs)
			setattr(self, dataSet+'_val', val_indexs)
			setattr(self, dataSet+'_train', train_indexs)

			setattr(self, dataSet+'_feats', feat_data)
			setattr(self, dataSet+'_labels', labels)
			setattr(self, dataSet+'_adj_lists', adj_lists)

		elif dataSet == 'pubmed':
			pubmed_content_file = self.config['file_path.pubmed_paper']
			pubmed_cite_file = self.config['file_path.pubmed_cites']

			feat_data = []
			labels = [] # label sequence of node
			node_map = {} # map node to Node_ID
			with open(pubmed_content_file) as fp:
				fp.readline()
				feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
				for i, line in enumerate(fp):
					info = line.split("\t")
					node_map[info[0]] = i
					labels.append(int(info[1].split("=")[1])-1)
					tmp_list = np.zeros(len(feat_map)-2)
					for word_info in info[2:-1]:
						word_info = word_info.split("=")
						tmp_list[feat_map[word_info[0]]] = float(word_info[1])
					feat_data.append(tmp_list)
			
			feat_data = np.asarray(feat_data)
			labels = np.asarray(labels, dtype=np.int64)
			
			adj_lists = defaultdict(set)
			with open(pubmed_cite_file) as fp:
				fp.readline()
				fp.readline()
				for line in fp:
					info = line.strip().split("\t")
					paper1 = node_map[info[1].split(":")[1]]
					paper2 = node_map[info[-1].split(":")[1]]
					adj_lists[paper1].add(paper2)
					adj_lists[paper2].add(paper1)
			
			assert len(feat_data) == len(labels) == len(adj_lists)
			test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

			setattr(self, dataSet+'_test', test_indexs)
			setattr(self, dataSet+'_val', val_indexs)
			setattr(self, dataSet+'_train', train_indexs)

			setattr(self, dataSet+'_feats', feat_data)
			setattr(self, dataSet+'_labels', labels)
			setattr(self, dataSet+'_adj_lists', adj_lists)


	def _split_data(self, num_nodes, test_split = 3, val_split = 6):
		rand_indices = np.random.permutation(num_nodes)

		test_size = num_nodes // test_split
		val_size = num_nodes // val_split
		train_size = num_nodes - (test_size + val_size)

		test_indexs = rand_indices[:test_size]
		val_indexs = rand_indices[test_size:(test_size+val_size)]
		train_indexs = rand_indices[(test_size+val_size):]
		
		return test_indexs, val_indexs, train_indexs


