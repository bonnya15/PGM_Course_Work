# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 19:48:09 2022

@author: shiuli Subhra Ghosh
"""

import time
start_time = time.time()

import numpy as np
import itertools

## Loading Data from structure.mat
from scipy.io import loadmat
s = loadmat('discrete_2000-revised.mat')
discrete = s['discrete_data']
M,N = discrete.shape
## Creating DAG matrix parent to chindren
dag = np.array([[0,0,1,1,0,1], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1], [0,0,0,0,0,0], [0,0,0,0,1,0]])


def parameter_learning(data, dag, alpha):
    M,N = data.shape
    node_size = np.array([max(discrete[:,i]) for i in range(N)])
    cpt_MLE = {}
    cpt_MAP = {}
    for i in range(N):
        parents = np.where(dag[:,i] == 1)[0]
        indexes = np.append(parents,i)
        n_parents = len(parents)
        pa_con = np.prod(node_size[parents])
        pa_size = node_size[parents]
        dim = np.append(node_size[parents],node_size[i])
        theta = np.zeros(dim)
        theta_MAP = np.zeros(dim)
        K = node_size[i]
        
        
        if pa_con == 1:
            node_con_list = list(range(1,node_size[i]+1))
            CPT = np.array([np.sum(data[:, i] == j)/M for j in node_con_list])
            cpt_MLE[i] = CPT
            cpt_MAP[i] = CPT
        else:
            alpha_i = alpha/(n_parents * node_size[i])
            temp = []
            temp1 = []
            for k in pa_size:
                temp.append(list(range(1,k+1)))
                temp1.append(list(range(k)))
            temp.append(list(range(1, node_size[i]+1)))
            temp1.append(list(range(node_size[i])))
            config = list(itertools.product(*temp))
            config1 = list(itertools.product(*temp1))
            for t in range(pa_con * node_size[i] ):
                new_array = np.where((discrete[:, indexes] == config[t]).all(axis = 1))[0]
                val = len(new_array)
                theta[config1[t]] = val 
                theta_MAP[config1[t]] = val + alpha_i - 1
            theta = theta.reshape(pa_con, node_size[i])
            theta_MAP = theta_MAP.reshape(pa_con, node_size[i])
            theta = theta/np.sum(theta, axis = 1).reshape(pa_con, 1) 
            theta_MAP = theta_MAP/(np.sum(theta_MAP, axis = 1).reshape(pa_con, 1) + ((K * alpha_i) - K))
            cpt_MLE[i] = theta
            cpt_MAP[i] = theta_MAP 
    return(cpt_MLE, cpt_MAP)
    

## Computing MLE and MAP CPTs where alpha = 5
cpt_MLE, cpt_MAP = parameter_learning(discrete, dag, 5)
        
print("--- %s seconds ---" % (time.time() - start_time))
    
        





