# ML-DOS : Machine Learning for studying Disordered Systems
Implementation of unsupervised GNN (GraphSAGE) to generate node representations for disordered systems. OPTICS clustering of obtained node embeddings reveals structure-dynamics correlation.


## "Unsupervised Graph Neural Network Reveals the Structure--Dynamics Correlation in Disordered Systems"

	-Vaibhav Bihani, Deparment of Civil Engineering, Indian Institute of Technology Delhi, Hauz Khas, New Delhi, India 110016
	-Sahil Manchanda,Department of Computer Science and Engineering,Indian Institute of Technology Delhi, Hauz Khas, New Delhi, India 110016
	-Sayan Ranu∗,Department of Computer Science and Engineering,Indian Institute of Technology Delhi, Hauz Khas, New Delhi, India 110016 and,Yardi School of Artificial Intelligence, Indian Institute of Technology Delhi, Hauz Khas, New Delhi, India 110016
	-N. M. Anoop Krishnan†, Department of Civil Engineering, Indian Institute of Technology Delhi, Hauz Khas, New Delhi, India 110016 and, Yardi School of Artificial Intelligence, Indian Institute of Technology Delhi, Hauz Khas, New Delhi, India 110016

The code for GraphSAGE GNN is a modified version of the original code by Tianwen Jiang (tjiang2@nd.edu), Tong Zhao (tzhao2@nd.edu), Daheng Wang (dwang8@nd.edu).
(https://github.com/twjiang/graphSAGE-pytorch)

# Contents

Folder 1 : Binary_lj_glass
    
    ->LAMMPS simulation script to generate LJ glass from melt-quench technique
    
    1)in.binary_loop	:main LAMMPS script
    2)pot.mod		  :potential file
    3)liq2.rest		  :liquid system datafile
    4)log.lammps		:simulation log
    5)run.cool		  :HPC job submission script

Folder : Graphs_dataset.zip
  
  ->160 graph structures(40 from each cooling rate trajectory), Cooled from T=2.0 to T=0.05, d_Temp=0.05 , indexed from [40*k , 40*(k+1)-1] for k=0,1,2,3 corresponding to four cooling rates from 3.33xe^-3 to 3.33xe^-6 

Folder : GSage_source_code
    
    ->Main GNN GraphSAGE code
    ->Contains files of a trial run of 10 epochs, for complete training more epochs required(>300)
    
    1)src			          :folder of source code
    2)Normals		        :folder of dataset of graphs of solid state Binary LJ glass(T<0.35)
    3)models		        :folder for storing models specified epochs
    4)PR_curves		      :Folder to save precision recall curves
    5)run.cool		      :HPC job submission script
    6)Fin_embs_*.csv	  :Final node embeddings 
    7)Fin_ext_*.csv		  :Node data of features specified in dataset(only node type and position is used in model) 
    8)Node_scores_*.csv	:Node_level performance metrics from random_walk test

Folder : Mean_squared_displacement
    
    ->LAMMPS simulation script to calculate MSD, and jupyter notebook to perform the analysis
    
    1)MSD Simulation	    :LAMMPS simulation script for mean squared displacement calculation
    2)MSD_analysis.ipynb	:Jupyter notebook for msd analysis
    3)LMPSTRajReader.py	  :Helper class to read LAMMPS trajectory file
    4)19Norm.gml		      :Graph data file
    5)Fin_embs_19.csv	    :Sample node embeddings data
	
Folder : Optics_clustering

    ->Jupyter notebook for optics clustering
    
    1)optics_clust.ipynb	:Jupyter notebook for OPTICS clustering and analysis
    2)Fin_embs_19.csv	    :Sample node embeddings data
    3)19Norm.gml		      :Graph data file

## Environment settings

- python==3.6.8
- pytorch==1.0.0




## Basic Usage

**Main Parameters:**

```
--dataSet     The input graph dataset. (default: cora)
--agg_func    The aggregate function. (default: Mean aggregater)
--epochs      Number of epochs. (default: 50)
--Val         Size of validation set(default: 10)
--b_sz        Batch size. (default: 20)
--seed        Random seed. (default: 824)
--unsup_loss  The loss function for unsupervised learning. ('margin' or 'normal', default: normal)
--config      Config file. (default: ./src/experiments.conf)
--cuda        Use GPU if declared.
```

**Learning Method**

The user can specify a learning method by --learn_method, 'sup' is for supervised learning, 'unsup' is for unsupervised learning, and 'plus_unsup' is for jointly learning the loss of supervised and unsupervised method.

**Example Usage**

To run the unsupervised model :
```
python -u -m src.main --epoch 50 --Val 5 --dataSet NormLJ --learn_method unsup
```


