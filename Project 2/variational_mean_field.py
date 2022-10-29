# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 00:56:18 2022

@author: shiuli Subhra Ghosh
"""

import numpy as np
## Loading Data from structure.mat

from scipy.io import loadmat
s = loadmat('structure.mat')
dag =s['dag']
domain_counts = s['domain_counts']

## Loading parameters of "CHILD BN" 
import pickle
with open('parameter.pkl', 'rb') as fp:
    CPTs = pickle.load(fp)

nodes = np.array(['BirthAsphyxia', 'Disease', 'Age', 'LVH', 'DuctFlow', 'CardiacMixing', 'LungParench',
    'LungFlow', 'Sick', 'HypDistrib', 'HypoxiaInO2', 'CO2', 'ChestXray', 'Grunting',
    'LVHreport', 'LowerBodyO2', 'RUQO2', 'CO2Report', 'XrayReport', 'GruntingReport'])
nodes_index = list(range(len(nodes)))

states = {key: list(range(domain_counts.reshape(-1)[key])) for key in range(len(nodes))}


## Initialization of B_nk
B_nk = {}
for i in range(len(domain_counts.reshape(-1))):
    B_nk[i] = np.array([np.random.uniform() for i in range(domain_counts.reshape(-1)[i])])
    B_nk[i] = B_nk[i] / np.sum(B_nk[i])

## Mean Field Algorithm
    
for i in nodes_index:
    for k in range(domain_counts.reshape(-1)[i]):
        