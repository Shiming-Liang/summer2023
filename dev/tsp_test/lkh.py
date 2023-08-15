#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:06:09 2023

@author: shiming
"""

import elkai
import numpy as np

cities = elkai.Coordinates2D({
    'city1': (0, 0),
    'city2': (0, 4),
    'city3': (5, 0)
})

print(cities.solve_tsp(runs=1))  # Output: ['city1', 'city2', 'city3', 'city1']

cities = elkai.DistanceMatrix([
    [0, 4, 0],
    [0, 0, 5],
    [0, 0, 0]
])

print(cities.solve_tsp(runs=1))  # Output: [0, 2, 1, 0]

cities = elkai.DistanceMatrix([
    [0, 0, np.Inf, 0],
    [0, 0, 0, 5],
    [np.Inf, 4, 0, 0],
    [0, 0, 0, 0]
])

print(cities.solve_tsp(runs=1))  # Output: [0, 2, 1, 0]
