# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 19:42:00 2022

@author: shiuli Subhra Ghosh
"""
from weighted_sampling import Weighted_Sampling
from gibbs_sampling import Gibbs_Sampling
import numpy as np
import matplotlib.pyplot as plt
## Loading Data from structure.mat

from scipy.io import loadmat
s = loadmat('structure.mat')
dag =s['dag']
domain_counts = s['domain_counts']

## Loading parameters of "CHILD BN" 
import pickle
with open('parameter.pkl', 'rb') as fp:
    CPTs = pickle.load(fp)

## Total Numbers of Samples to be selected
T = 50000

## Setting the nodes in topological ordering

nodes = np.array(['BirthAsphyxia', 'Disease', 'Age', 'LVH', 'DuctFlow', 'CardiacMixing', 'LungParench',
    'LungFlow', 'Sick', 'HypDistrib', 'HypoxiaInO2', 'CO2', 'ChestXray', 'Grunting',
    'LVHreport', 'LowerBodyO2', 'RUQO2', 'CO2Report', 'XrayReport', 'GruntingReport'])
Evidence = ['CO2Report','LVHreport','XrayReport']
Evidence_values = [0,0,2]


#####################
## Weighted Sampling 
#####################

Query = ['BirthAsphyxia']
Query_val = 0

inference = Weighted_Sampling(T, nodes,dag, CPTs, domain_counts, Evidence, Evidence_values, Query, Query_val)
infer,samples, weights = inference.PPE()
print("P('BirthAsphyxia' = 'yes' | 'CO2Report'= '<7.5', 'LVHreport' = 'yes', 'XrayReport' = 'Plethoric') = {}".format(infer))


## MAP Inference
Query = ['Disease']
Query_val = 0
inference = Weighted_Sampling(T, nodes,dag, CPTs, domain_counts, Evidence, Evidence_values, Query, Query_val)
MAP = inference.MAP(samples,weights)
print("argmax('Disease' | 'CO2Report'= '<7.5', 'LVHreport' = 'yes', 'XrayReport' = 'Plethoric') = {})".format(MAP))

#####################
## Gibbs Sampling
#####################

## Total Numbers of Samples to be selected

T = 10000000

Query = ['BirthAsphyxia']
Query_val = 0
burn_in = 10000
step_size = 50

gibbs = Gibbs_Sampling(T, nodes,dag, CPTs, domain_counts, Evidence, Evidence_values)
total_samples = gibbs.Sample_Generation()
infer = gibbs.PPE(burn_in, step_size, Query, Query_val)
print("P('BirthAsphyxia' = 'yes' | 'CO2Report'= '<7.5', 'LVHreport' = 'yes', 'XrayReport' = 'Plethoric') = {}".format(infer))

## MAP Inference 

Query = ['Disease']
Query_val = 0
MAP = gibbs.MAP(burn_in, step_size, Query, Query_val)
print("argmax('Disease' | 'CO2Report'= '<7.5', 'LVHreport' = 'yes', 'XrayReport' = 'Plethoric') = {})".format(MAP))





