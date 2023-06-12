#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:25:54 2023

@author: shiming
"""

import numpy as np
from matplotlib import pyplot as plt

toy_dataset = np.array([[0, 0, 0, 0, 0],
                        [3, 4, 3, 10, 0],
                        [4, 3, 3, 10, 0],
                        [5, 5, 5, 10, 0],
                        [1, 1, 1, 10, 0],
                        [2, 2, 2, 20, 0],
                        [2, 2, 2, 20, 2],
                        [2, 2, 2, 20, -2],
                        [0, 0, 0, 30, 0]])

plt.figure()
plt.scatter(toy_dataset[0, 3], toy_dataset[0, 4], c='r')
plt.scatter(toy_dataset[1:-1, 3], toy_dataset[1:-1, 4], c='y')
plt.scatter(toy_dataset[-1, 3], toy_dataset[-1, 4], c='b')
plt.xlabel('long')
plt.ylabel('lat')
plt.title('toy dataset')
plt.savefig('toy_dataset.jpg')
