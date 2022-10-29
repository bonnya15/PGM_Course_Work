# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:14:16 2022

@author: shiuli Subhra Ghosh
"""

import numpy as np

class Gibbs_Sampling:
    def __init__(self, T, nodes,dag, CPTs, domain_counts, Evidence, Evidence_values):
        self.T = T 
        self.nodes = nodes
        self.CPTs = CPTs
        self.Evidence = Evidence
        self.domain_counts = domain_counts
        self.Evidence_values = Evidence_values
        self.dag = dag
        self.Evidence_index = []
        for i in self.Evidence:
            self.Evidence_index.append(np.where(self.nodes == i)[0][0])
        self.Evidence_dict = dict(zip(self.Evidence_index, self.Evidence_values))
        self.node_dict = dict(zip(list(range(len(self.nodes))), self.nodes))
        self.samples = np.zeros((self.T,len(self.nodes)))
        
        
    @staticmethod    
    def cumsum(lists):
        cu_list = []
        length = len(lists)
        cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)]
        return np.array(cu_list[1:])
    
    def sample_initialization(self):
        initial_sample = []
        for i, node in enumerate(self.nodes):
            if i not in self.Evidence_dict.keys():
                initial_sample.append(np.random.randint(self.domain_counts.reshape(-1)[i]))
            else:
                initial_sample.append(self.Evidence_dict[i])
        return(initial_sample)
            
    def Sample_Generation(self):
        initial_sample = np.array(Gibbs_Sampling.sample_initialization(self))
        self.samples[0,:] = initial_sample.reshape(1,20)
        nodes_index = np.array(range(len(self.nodes)))
        for t in range(self.T - 1):
            current_sample = self.samples[t,:]
            random_index = np.random.randint(len(self.nodes))
            if random_index not in self.Evidence_dict.keys():
                parents = nodes_index[self.dag[:,random_index] == 1]
                children = nodes_index[self.dag[random_index,:] == 1]
                parents_val = current_sample[parents]
                if len(children) != 0:
                    w = 1
                    for i in children:
                        parents_index = nodes_index[self.dag[:,i] == 1]
                        children_parents_values = current_sample[parents_index]
                        pa_cardinality = list(self.domain_counts.reshape(-1)[parents_index])
                        pa_cardinality.insert(0, self.domain_counts.reshape(-1)[i])
                        cpt = self.CPTs[self.node_dict[i]].reshape(pa_cardinality)
                        if len(parents_index) == 1: ## Because if the child has only one parent that is the actual node
                            w = w * cpt[int(self.samples[t][i]), :]
                        elif len(children_parents_values) == 2:
                            g = np.where(parents_index == random_index)[0][0]
                            if g == 0:
                                w = w * cpt[int(self.samples[t][i]), : , int(children_parents_values[1])]
                            elif g == 1:
                                w = w * cpt[int(self.samples[t][i]), int(children_parents_values[0]) , :]
                                
                    
                else:
                    w = 1
                #print(t, len(children), w)
                if len(parents) != 0:
                    cardinality = list(self.domain_counts.reshape(-1)[parents])
                    cardinality.insert(0, self.domain_counts.reshape(-1)[random_index])
                    cpt = self.CPTs[self.node_dict[random_index]].reshape(cardinality)
                    if len(parents) == 1:
                        p = cpt[:,int(parents_val[0])]
                    elif len(parents) == 2:
                        p = cpt[:, int(parents_val[0]), int(parents_val[1])]
                else:
                    p = self.CPTs[self.node_dict[random_index]]
                prob = p.reshape(-1) * w
                cum = Gibbs_Sampling.cumsum(prob) / np.sum(prob)
                random = np.random.random_sample()
                ind = np.where(cum > random)[0][0]
                self.samples[t+1] = self.samples[t]
                self.samples[t+1][random_index] = ind
            else:
                self.samples[t+1] = self.samples[t]
            print("Iterations Completed : {}".format(t))   
        return(self.samples)
        
    def PPE(self, burn_in, step_size, Query, Query_val):
        query_index = np.where(self.nodes == Query)[0][0]
        new_sample_wo_burn_in = self.samples[burn_in: ,:].copy()         
        steps = np.arange(0, self.T-burn_in, step_size)              
        new_samples = new_sample_wo_burn_in[steps]  
        len_yes = np.where(new_samples[:,query_index] == Query_val)[0].shape[0]
        total = new_samples.shape[0]
        inference = len_yes / total 
        return(inference)
        
    def MAP(self, burn_in, step_size,  Query, Query_val):
        query_index = np.where(self.nodes == Query)[0][0]
        new_sample_wo_burn_in = self.samples[burn_in: ,:].copy()         
        steps = np.arange(0, self.T-burn_in, step_size)              
        new_samples = new_sample_wo_burn_in[steps] 
        cardinality = self.domain_counts.reshape(-1)[query_index]
        total = new_samples.shape[0]
        MAP_array = []
        for i in range(cardinality):
            len_yes = np.where(new_samples[:,query_index] == i)[0].shape[0]               
            map_inference = len_yes / total
            MAP_array.append(map_inference)                
        max_index = np.argmax(np.array(MAP_array)) 
        return(max_index)
            
        
        
        




































