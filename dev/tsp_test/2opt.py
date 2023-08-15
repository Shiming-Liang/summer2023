#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:05:06 2023

@author: shiming
"""

from py2opt.routefinder import RouteFinder
import numpy as np

# cities_names = ['A', 'B', 'C', 'D']
# dist_mat = [[0, 29, 15, 35], [29, 0, 57, 42],
#             [15, 57, 0, 61], [35, 42, 61, 0]]
# route_finder = RouteFinder(dist_mat, cities_names, iterations=10)
# best_distance, best_route = route_finder.solve()

# print(best_distance)
# print(best_route)

cities_names = ['A', 'B', 'C', 'D']
dist_mat = [
    [0, 0, np.inf, 0],
    [0, 0, 0.01, 5],
    [np.inf, 4, 0, 0.01],
    [0, 0.01, 0.01, 0]
]
route_finder = RouteFinder(dist_mat, cities_names,
                           iterations=1, return_to_begin=True, verbose=False)
best_distance, best_route = route_finder.solve()

print(best_distance)
print(best_route)
