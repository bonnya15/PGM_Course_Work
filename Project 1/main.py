# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 13:23:33 2022

@author: shiuli Subhra Ghosh
"""
##%load_ext autoreload
##%autoreload 2

import numpy as np
from cal import Node
from functools import reduce

##### Class problem 


####################################################

## Building S
S = Node("S")
S.cardinality = 2
S.priors = np.array([0.2,0.8])
S.likelihood = np.array([1,1])  ## P(S=1) = 0.8


## Building A
A = Node("A")
A.cardinality = 2
A.priors = np.array([0.4, 0.6]) ## P(A=1) = 0.6
A.likelihood = np.array([1,1])  


## Building B|S
temp = np.zeros((2,2))
temp[0,0] = 0.7
temp[0,1] = 0.3
temp[1,0] = 0.3
temp[1,1] = 0.7
B = Node("B")
B.cardinality = 2
B.m = temp
B.likelihood = np.array([1, 1])
B.priors = np.array([1, 1])

## Building L|S  ## Evidence Node
temp = np.zeros((2,2))
temp[0,0] = 0.7
temp[0,1] = 0.2
temp[1,0] = 0.3
temp[1,1] = 0.8
L = Node("L")
L.cardinality = 2
L.m = temp
L.likelihood = np.array([0, 1])
L.priors = np.array([0, 1])

## Building T|A
temp = np.zeros((2,2))
temp[0,0] = 0.7
temp[0,1] = 0.2
temp[1,0] = 0.3
temp[1,1] = 0.8
T = Node("T")
T.cardinality = 2
T.m = temp
T.likelihood = np.array([1, 1])
T.priors = np.array([1, 1])


# Building E|L,T 
E = Node("E")
E.cardinality = 2
m = np.zeros((2,2,2))
m[0,0,0] = 1 - 0.1
m[0,0,1] = 1 - 0.7
m[0,1,0] = 1 - 0.6
m[0,1,1] = 1 - 0.8
m[1,0,0] = 0.1
m[1,0,1] = 0.7
m[1,1,0] = 0.6
m[1,1,1] = 0.8
E.m = m 
E.likelihood = np.array([1, 1])
E.priors = np.array([1, 1])

## Building X|E  ## Evidence Node
temp = np.zeros((2,2))
temp[0,0] = 0.9
temp[0,1] = 0.2
temp[1,0] = 0.1
temp[1,1] = 0.8
X = Node("X")
X.cardinality = 2
X.m = temp
X.likelihood = np.array([1, 0])
X.priors = np.array([1, 0])

## Building F 
F = Node("F")
F.cardinality = 2
m = np.zeros((2,2,2))
m[0,0,0] = 1 - 0.2
m[0,0,1] = 1 - 0.8
m[0,1,0] = 1 - 0.7
m[0,1,1] = 1 - 0.9
m[1,0,0] = 0.2
m[1,0,1] = 0.8
m[1,1,0] = 0.7
m[1,1,1] = 0.9
F.m = m 
F.likelihood = np.array([1, 1])

## Creating the graph and adding dependencies
L.add_parent(S)
B.add_parent(S)
E.add_parent(L)
E.add_parent(T)
T.add_parent(A)
F.add_parent(B)
F.add_parent(E)
X.add_parent(E)

evidence_nodes = [L,X]
list_nodes = [B,E,A,F,S,T]  ## List of non-evidence nodes


## Need to calculate P()

iterations = 10
i = 0 ## Initializing the counter for iterations
iteration_belief_array = []
while i <= iterations:
    belief_array = []
    for k in list_nodes:
        parents = k.parents
        children = k.children
        print("----------------------------------")
        print("Current Node is {}".format(k.name))
        print("----------------------------------")
        ## Step - 1 Sending messages from Parents
        for p in parents:
            m2c = p.message_to_child(k)
            print("message from {} to {} = {}".format(p,k,m2c))
        ## Step - 2 Computing total messages from Parents
        if len(parents) != 0:    
            total_msg_from_parent = k.total_msg_from_parent()
            print("total message from parents to {} = {}".format(k, total_msg_from_parent))
        ## Step - 3 Collecting messages from children
        for c in children:
            c2k = c.message_to_parent(k)
            print("message from {} to {} = {}".format(c,k, c2k))
        ## Computing total messages from children
        if len(children) != 0:
            total_msg_from_child = k.total_msg_from_child()
            print("total message from children to {} = {}".format(k, total_msg_from_child))
        ## Updating belief for non-evidence nodes
        updated_belief = k.get_belief()
        print("updated belief for {} = {}".format(k, updated_belief))
        belief_array.append(updated_belief.reshape(-1))
    iteration_belief_array.append(np.array(belief_array))  
    ## Convergence Ceiterion
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
    




