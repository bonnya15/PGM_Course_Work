# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 00:31:36 2022

@author: shiuli Subhra Ghosh
"""

import time
start_time = time.time()


import numpy as np
from main import parameter_learning
import matplotlib.pyplot as plt

## Loading Data from structure.mat
from scipy.io import loadmat
s = loadmat('discrete_2000-revised.mat')
discrete = s['discrete_data']
M,N = discrete.shape
## Creating DAG matrix parent to chindren
dag = np.array([[0,0,1,1,0,1], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1], [0,0,0,0,0,0], [0,0,0,0,1,0]])

cpt_MLE, cpt_MAP = parameter_learning(discrete, dag, 5)

## Function for finding joint probability

def joint_prob(cpt):
    A = np.reshape(cpt[0], (2, 1, 1, 1, 1, 1,))
    B = np.reshape(cpt[1], (1, 3, 1, 1, 1, 1,))
    C = np.moveaxis(np.reshape(cpt[2], (2, 3, 1, 1, 1, 1,)), [0, 1], [0, 2])
    D = np.moveaxis(np.reshape(cpt[3], (2, 3, 2, 1, 1, 1,)), [0, 1, 2], [0, 1, 3])
    E = np.moveaxis(np.reshape(cpt[4], (3, 2, 2, 1, 1, 1,)), [0, 1, 2], [2, 5, 4])
    F = np.moveaxis(np.reshape(cpt[5], (2, 2, 2, 1, 1, 1,)), [0, 1, 2], [0, 3, 5])
    
    joint = A * B * C * D * E * F
    
    return(joint)

## True probability table for finding true joint 

A = np.reshape(np.array([0.4,0.6]), (2, 1, 1, 1, 1, 1,))
B = np.reshape(np.array([0.3,0.4,0.3]), (1, 3, 1, 1, 1, 1,))
C = np.moveaxis(np.reshape(np.array([[0.3, 0.25, 0.45], [0.05, 0.2, 0.75]]), (2, 3, 1, 1, 1, 1,)), [0, 1], [0, 2])
D = np.moveaxis(np.reshape(np.array([[0.05, 0.95], [0.93,0.07], [0.41,0.59], [0.22,0.78], [0.53,0.47], [0.3,0.7]]), (2, 3, 2, 1, 1, 1,)), [0, 1, 2], [0, 1, 3])
E = np.moveaxis(np.reshape(np.array([[0.02,0.98], [0.32,0.68], [0.3,0.7], [0.92, 0.08], [0.13, 0.87], [0.58, 0.42]]), (3, 2, 2, 1, 1, 1,)), [0, 1, 2], [2, 5, 4])
F = np.moveaxis(np.reshape(np.array([[0.95, 0.05], [0.1, 0.9], [0.43, 0.57], [0.28, 0.72]]), (2, 2, 2, 1, 1, 1,)), [0, 1, 2], [0, 3, 5])

true_joint = A * B * C * D * E * F


## KL Divergence function 
  
def kl_divergence(a, b):
    return(a * np.log(a/b)).sum()

kl_divergence(true_joint, joint_prob(cpt_MAP))
kl_divergence(true_joint, joint_prob(cpt_MLE))

## Sampling from dataset
n = np.arange(10,2000,50)
alpha = np.array([5,10,20,50])
kl = {}

for j in alpha:
    temp = []
    MLE = []
    for i in n:
        random_indices = np.random.choice(2000, size=i, replace=False)
        new = discrete[random_indices, :]
        cpt_MLE, cpt_MAP = parameter_learning(new, dag, j)
        kl_div_MAP = kl_divergence(true_joint, joint_prob(cpt_MAP))
        kl_div_MLE = kl_divergence(true_joint,joint_prob(cpt_MLE))
        temp.append(kl_div_MAP)
        MLE.append(kl_div_MLE)
    kl[j] = temp
        
## Plotting KL divergence with sample size and for different alpha values. 
        
plt.plot(n, kl[5], label = 'alpha = 5')
plt.plot(n, kl[10], label = 'alpha = 10')
plt.plot(n, kl[20], label = 'alpha = 20')
plt.plot(n, kl[50], label = 'alpha = 50')
plt.plot(n, MLE, label = 'MLE')
plt.legend()
plt.xlabel('No. of Samples')
plt.ylabel('KL Divergence')
#plt.savefig("KL_divergence.png", dpi=1200)
plt.show()

            
print("--- %s seconds ---" % (time.time() - start_time))








        