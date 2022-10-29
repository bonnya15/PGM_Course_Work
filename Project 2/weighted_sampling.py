# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 19:19:07 2022

@author: shiuli Subhra Ghosh
"""
import numpy as np


class Weighted_Sampling:
    def __init__(self, T, nodes,dag, CPTs, domain_counts, Evidence, Evidence_values, Query, Query_val):
        self.T = T 
        self.nodes = nodes
        self.CPTs = CPTs
        self.Evidence = Evidence
        self.Query = Query
        self.domain_counts = domain_counts
        self.Evidence_values = Evidence_values
        self.dag = dag
        self.Query_val = Query_val
        

    
    @staticmethod    
    def cumsum(lists):
        cu_list = []
        length = len(lists)
        cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)]
        return np.array(cu_list[1:])    
    
    def PPE(self):
        #nodes_index = list(range(len(self.nodes)))
        #states = {key: list(range(self.domain_counts.reshape(-1)[key])) for key in range(len(self.nodes))}
        weights = np.ones(self.T)
        samples = np.zeros((self.T,len(self.nodes)))
        Evidence_index = []
        for i in self.Evidence:
            Evidence_index.append(np.where(self.nodes == i)[0][0])
        Evidence_dict = dict(zip(Evidence_index, self.Evidence_values))
        
        for t in range(self.T):
            weight = 1
            for i, node in enumerate(self.nodes):
                parents = self.nodes[self.dag[:,i]==1]
                parents_index = [np.where(self.nodes == i)[0][0] for i in parents]
                if i not in Evidence_index:
                    if len(parents) == 0:
                        random = np.random.random_sample()
                        #cardinality = domain_counts.reshape(-1)[i]
                        cpt = self.CPTs[node].reshape(-1)
                        cum = Weighted_Sampling.cumsum(cpt)
                        ind = np.where(cum > random)[0][0] ## Random Sampling 
                        samples[t][i] = ind
                    else:
                        current_samples = samples[t]
                        parent_values = current_samples[parents_index]
                        pa_cardinality = list(self.domain_counts.reshape(-1)[parents_index])
                        pa_cardinality.insert(0, self.domain_counts.reshape(-1)[i])
                        cpt = self.CPTs[node].reshape(pa_cardinality)
                        if len(parent_values) == 1:
                           choose = cpt[:,int(parent_values[0])]
                        elif len(parent_values) == 2:
                            choose = cpt[:, int(parent_values[0]), int(parent_values[1])]
                        cum = Weighted_Sampling.cumsum(choose)
                        random = np.random.random_sample()
                        ind = np.where(cum > random)[0][0]
                        samples[t][i] = ind
                else:
                    samples[t][i] = Evidence_dict[i]
                    current_samples = samples[t]
                    parent_values = current_samples[parents_index]
                    pa_cardinality = list(self.domain_counts.reshape(-1)[parents_index])
                    pa_cardinality.insert(0, self.domain_counts.reshape(-1)[i])
                    cpt = self.CPTs[node].reshape(pa_cardinality)
                    if len(parent_values) == 1:
                        w = cpt[int(samples[t][i]),int(parent_values[0])]
                    elif len(parent_values) == 2:
                        w = cpt[int(samples[t][i]), int(parent_values[0]), int(parent_values[1])]
                    weight = weight * w
            weights[t] = weight        
        
        total = np.sum(weights)  
        query_index = np.where(self.nodes == self.Query)[0][0]
        query_weights_0 = weights[samples[:,query_index] == self.Query_val]                
        inference = np.sum(query_weights_0) / total   
        return(inference,samples, weights)
        
    def MAP(self, samples,weights):
        total = np.sum(weights) 
        query_index = np.where(self.nodes == self.Query)[0][0]
        cardinality = self.domain_counts.reshape(-1)[query_index]
        MAP_array = []
        for i in range(cardinality):
            query_weights = weights[samples[:,query_index] == i]                
            map_inference = np.sum(query_weights) / total
            MAP_array.append(map_inference)
                
        max_index = np.argmax(np.array(MAP_array)) 
        return(max_index)
    