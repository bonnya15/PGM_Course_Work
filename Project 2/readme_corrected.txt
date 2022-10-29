------------------------------------------------------------------------------------------------------------------------------
1.to get the structure of the CHILD BN,  you need "structure.mat" file
a) to open it in Matlab: 
       load structure.mat
b) to open it in python: 
       from scipy.io import loadmat
       s = loadmat('structure.mat')
       dag = s['dag']
       domain_counts = s['domain_counts']
c) after you open it, you should have two variables: 'dag' and 'domain_count'
d) 'dag' is a 20x20 matrix showing structure relationships between each node.  For each row, if there is a column that has value 1, it is saying that the node of the same column index is the child of the node of the same row index.
For example, If we have value 1 at location (row = 9, col = 3), this means node 9 is the parent of node 3, in other words, node 3 is the child of node 9.
e) 'domain_count' is a 1x20 array showing the number of states for each node.

--------------------------------------------------------------------------------------------------------------------------------
2. to get parameters of CHILD BN, you can use "parameter.pkl" or "parameter.mat":
a) "parameter.pkl": you can open it in python with following codes
          import pickle
          with open('parameter.pkl', 'rb') as fp:
                  CPTs = pickle.load(fp)

b) "parameter.mat": you can open it in Matlab with
          load parameter.mat

c) after you open it, you should have 20 CPTs that are conditional probability tables for 20 nodes.  

d) In each CPT,  rows are corresponding to the states of the node, in the same order as we list in the introduction file. 
For example, if the node is of binary state:{"yes", "no"},  then the first row is corresponding to state "yes" and the second row is corresponding to state "no".

e) In each CPT, columns are corresponding to the states of the condition nodes.
If the node has only one parent, then the order of the columns will be exactly the same with the order of states of that parent as we listed.
For example, the CPT(the 2 by 6 matrix you have) for node 4(LVH) should be unserdood as:

LVH                                    Disease(node 2)
(node 4)
	 PFC	TGA	Fallot	PAIVS	TAPVD	Lung
yes	 0.1	0.1	0.1	0.9	0.05	0.1
no	 0.9	0.9	0.9	0.1	0.95	0.9

f) There are six nodes that are having more than one parent:
    P(Age:3 | Disease:6, Sick:2)
    P(HypDistrib:2 | DuctFlow:3, CardiacMixing:4)
    P(HypoxiaInO2:3 | CardiacMixing:4, LungParench:3)
    P(ChestXray:5 | LungParench: 3, LungFlow: 3)
    P(Grunting: 2 | LungParench: 3, Sick: 2)
    P(LowerBodyO2:3 | HypDistrib:2, HypoxiaInO2:3)

We will first order the states of the second parent and then order the states of the first parent. Take "P(LowerBodyO2 | HypDistrib, HypoxiaInO2)" as the example.

Given parents' states:
Hypoxia distribution: {??equal??, ??unequal??} 
Hypoxia in O2: {??None??, ??Moderate??, ??Severe??}

the 3 by 6 CPT for node 16 should be understood as:

LowerBody O2                                  
(node 16)
	                                {"equal","None"}	{"equal","Moderate"}	          {"equal","severe"}         {"unequal","None"}	{"unequal","Moderate"}       {"unequal","severe"}
<5	                                            0.1                                   0.4                                         0.3                                     0.5                                      0.5                                         0.6
5-12	                                            0.3                                   0.5                                         0.6                                     0.45                                    0.4                                         0.35
12+                                                       0.6                                    0.1                                        0.1                                     0.05                                     0.1                                         0.05
