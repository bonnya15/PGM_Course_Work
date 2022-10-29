# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 01:06:37 2022

@author: shiuli Subhra Ghosh
"""

import numpy as np
import matplotlib.pyplot as plt
from gibbs_sampling import Gibbs_Sampling

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

#######################
## Visualization 
#######################

from scipy.io import loadmat
total_samples = loadmat('C:\\Users\\shiuli Subhra Ghosh\\Box\\Documents\\gibbs_samples.mat')
total_samples = total_samples['total_samples']
 

## Effect of burn_in period when the sample is 10000000
## burn_in time = [0,50000], step_size = 50 (fixed)


def PPE(samples,T, burn_in, step_size, Query, Query_val):
    nodes = np.array(['BirthAsphyxia', 'Disease', 'Age', 'LVH', 'DuctFlow', 'CardiacMixing', 'LungParench',
    'LungFlow', 'Sick', 'HypDistrib', 'HypoxiaInO2', 'CO2', 'ChestXray', 'Grunting',
    'LVHreport', 'LowerBodyO2', 'RUQO2', 'CO2Report', 'XrayReport', 'GruntingReport'])
    samples = samples[0:T,:]
    query_index = np.where(nodes == Query)[0][0]
    new_sample_wo_burn_in = samples[burn_in: ,:].copy()         
    steps = np.arange(0, T-burn_in, step_size)              
    new_samples = new_sample_wo_burn_in[steps]  
    len_yes = np.where(new_samples[:,query_index] == Query_val)[0].shape[0]
    total = new_samples.shape[0]
    inference = len_yes / total 
    return(inference)

T = 10000000
burn_in_array = np.arange(0,50000,100)
step_size = 50
Query = ['BirthAsphyxia']
Query_val = 0
inference_array = []
for i in burn_in_array:
    inference_array.append(PPE(total_samples, T, i, step_size, Query, Query_val))
    print('iterations = {}'.format(i))

plt.plot(burn_in_array, inference_array)
plt.xlabel("burn_in time")
plt.ylabel("P(BirthAsphyxia | Evidence)")
plt.show()

## Effect of step_size when the sample is 10000000
## step_size = [0,100], burn_in = 10000 (fixed)

T = 100000
step_size_array = np.arange(0,100,5)
burn_in = 10000
Query = ['BirthAsphyxia']
Query_val = 0
inference_array_step = []
for i in step_size_array:
    inference_array_step.append(PPE(total_samples, T, i, step_size, Query, Query_val))
    print('iterations = {}'.format(i))


plt.plot(step_size_array, inference_array_step,'o-')
plt.xlabel("step_size / skip_time")
plt.ylabel("P(BirthAsphyxia | Evidence)")
plt.show()

## Effect of Initialization 
#  10 initializations

T = 100000

Query = ['BirthAsphyxia']
Query_val = 0
burn_in = 10000
step_size = 20
initialization_array = []

for i in range(10):
    gibbs = Gibbs_Sampling(T, nodes,dag, CPTs, domain_counts, Evidence, Evidence_values)
    total_samples = gibbs.Sample_Generation()
    infer = gibbs.PPE(burn_in, step_size, Query, Query_val)
    initialization_array.append(infer)

l = [i for i in range(10)]

plt.plot(l, initialization_array,'o-')
plt.xlabel("Random Initialization")
plt.ylabel("P(BirthAsphyxia | Evidence)")
plt.show()




