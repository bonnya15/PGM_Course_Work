# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 13:23:33 2022

@author: shiuli Subhra Ghosh
"""
##%load_ext autoreload
##%autoreload 2

import numpy as np
from cal1 import Node
from functools import reduce


B = Node("B")
B.cardinality = 2
m = np.zeros((1,2))
m[0,0] = 0.001
m[0,1] = 0.999
B.priors = m
B.likelihood = np.array([1,1])

E = Node("E")
E.cardinality = 2
m = np.zeros((1,2))
m[0,0] = 0.002
m[0,1] = 0.998
E.priors = m
E.likelihood = np.array([1,1])

P = Node("P")
P.cardinality = 2
m = np.zeros((1,2))
m[0,0] = 0.05
m[0,1] = 0.95
P.priors = m
P.likelihood = np.array([1,1])

M = Node("M")
M.cardinality = 2
M.m = np.array([[0.7, 0.01 ],[0.3, 0.99]])
M.likelihood = np.array([1,0])
M.priors = np.array([1,0])

A = Node("A")
A.cardinality = 2
m = np.zeros((2,2,2))
m[0,0,0] = 0.95
m[0,0,1] = 0.94
m[0,1,0] = 0.29
m[0,1,1] = 0.001
m[1,0,0] = 1 - 0.95
m[1,0,1] = 1 - 0.94
m[1,1,0] = 1 - 0.29
m[1,1,1] = 1 - 0.001
A.m = m 

J = Node("J")
J.cardinality = 2
m = np.zeros((2,2,2))
m[0,0,0] = 0.95
m[0,0,1] = 0.5
m[0,1,0] = 0.90
m[0,1,1] = 0.01
m[1,0,0] = 1 - 0.95
m[1,0,1] = 1 - 0.5
m[1,1,0] = 1 - 0.90
m[1,1,1] = 1 - 0.01
J.m = m 
J.likelihood = np.array([1,1])



A.add_parent(B)
A.add_parent(E)
J.add_parent(P)
J.add_parent(A)
M.add_parent(A)

evidence_nodes = [M]
list_nodes = [A,B,E,J,P]

## Need to calculate P()

iterations = 3
i = 0 ## Initializing the counter for iterations
iteration_belief_array = []
while i <= iterations:
    belief_array = []
    for k in list_nodes:
        parents = k.parents
        children = k.children
        for p in parents:
            p.message_to_child(k)
            print("message from {} to {}".format(p,k))
        if len(parents) != 0:    
            total_msg_from_parent = k.total_msg_from_parent()
        print("total message from parents to {}".format(k))
        for c in children:
            c.message_to_parent(k)
            print("message from {} to {}".format(c,k))
        if len(children) != 0:
            total_msg_from_child = k.total_msg_from_child()
        updated_belief = k.get_belief()
        print("updated belief for {}".format(k))
        belief_array.append(updated_belief.reshape(-1))
    iteration_belief_array.append(np.array(belief_array))    
    if (i >= 1) & ((iteration_belief_array[i-1] - iteration_belief_array[i]).all() < 10**(-6)):
        print("Iterations terminated in {}th step".format(i+1))
        print("##############################################")
        for i in range(len(list_nodes)):
            print("P({} = 1) | L = 1, X = 0 = {}".format(list_nodes[i].name, belief_array[i][1]))
        print("##############################################")
        break
    else:
        i = i +1
        continue
    
    
        
        
        
    
        




