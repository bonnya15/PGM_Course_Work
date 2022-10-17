import numpy as np
from functools import reduce


class Node:
    def __init__(self, name):
        self.name = name
        self.cardinality = None
        self.likelihood = None
        self.priors = None
        self.belief = None
        self.parents = []
        self.children = []
        self.m = None

    def add_parent(self, node):
        self.parents.append(node)
        node.children.append(self)

    def __str__(self):
        return self.name

    def message_to_parent(self, parent):
        """
        returns marginalized out parent message:
            - in m: group all entries by receiver parent values (all with 0 together, all with 1 together)
            - use other values in groups to get likelihood and messages from other parents
            - multiply those values in each group element
            - sum each group
        """
        if self.likelihood is not None:
            likelihood = self.likelihood
        parents_priors = np.array([p.message_to_child(self) for p in self.parents if p != parent])
        temp = np.zeros((1, parent.cardinality))
        parent_i = self.parents.index(parent)
        if parent_i == 0:
            m = self.m
        if parent_i == 1:
            m = self.m.swapaxes(parent_i, parent_i +1)
        if len(parents_priors) == 0:
            for i in range(parent.cardinality):
                flag = 0
                for j in range(self.cardinality):
                    flag += likelihood[j] * self.m[j][i]
                temp[0,i] = flag
            return(temp)
        
        if len(parents_priors) == 1:
            x = parents_priors[0]
        elif len(parents_priors) > 1:
            x = np.dot(parents_priors[0].T, parents_priors[1])

        for i in range(parent.cardinality):
            flag = 0
            for j in range(self.cardinality):
                flag += likelihood[j] * np.sum(m[j][i] * x) 
            temp[0,i] = flag
        return(temp)
    
    def total_msg_from_child(self):
        incoming_children_messages = np.array([c.message_to_parent(self) for c in self.children])
        self.likelihood = incoming_children_messages.prod(axis=0).reshape(2,)
        return(self.likelihood)
        

    def message_to_child(self, child):
        children_messages = np.array([c.message_to_parent(self) for c in self.children if c != child])
        if (len(children_messages) > 0):
            unnormalized = (children_messages * self.priors.reshape(1,2)).prod(axis=0)
            return unnormalized
        return self.priors.reshape(1,2)


    def total_msg_from_parent(self):
        parents_messages = [p.message_to_child(self) for p in self.parents]
        if len(parents_messages) != 0:
            temp = np.zeros((1, self.cardinality))
            if len(parents_messages) == 1:
                x = parents_messages[0]
            else:
                x = np.dot(parents_messages[0].T, parents_messages[1])
            for i in range(self.cardinality):
                temp[0,i] = np.sum(self.m[i] * x)
            self.priors = temp
            return temp
        else:
            return(self.priors.reshape(1,2))
        

    def get_belief(self):
        unnormalized = self.likelihood * self.priors
        self.belief = unnormalized/unnormalized.sum()
        return(self.belief)
    

 
    
    
