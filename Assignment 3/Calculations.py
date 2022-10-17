# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 10:54:00 2022

@author: shiuli Subhra Ghosh
"""

import numpy as np

a = np.array([0.133, 0.007, 0.42, 0.14, 0.057, 0.003, 0.18, 0.06])
b = np.array([0.5529, 0.0291, 0.291, 0.097, 0.0171, 0.0009, 0.009, 0.003])
a = a.reshape(1,8)
b = b.reshape(1,8)

c = 0.8*a + 0.2*b

sum(a[:,0:4].reshape(-1))
sum(a[:,4:].reshape(-1))
result_1 = 0.02*sum(a[:,0:4].reshape(-1)) + 0.6 * sum(a[:,4:].reshape(-1))


sum(b[:,0:4].reshape(-1))
sum(b[:,4:].reshape(-1))
result_2 = 0.02*sum(b[:,0:4].reshape(-1)) + 0.6 * sum(b[:,4:].reshape(-1))

unnormalized = np.array([0.8*result_1 , 0.2*result_2])

normalized = unnormalized/unnormalized.sum()

unnormalized = np.array([0.05089, 0.3956])