#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:50:17 2023

@author: shiming
"""

import networkx as nx
import networkx.algorithms.approximation as approx

G = nx.complete_graph(3, create_using=nx.DiGraph)
nx.set_edge_attributes(
    G, {(0, 1): 2, (1, 2): 2, (2, 0): 2, (0, 2): 1, (2, 1): 1, (1, 0): 1}, "weight")
tour = approx.asadpour_atsp(G, source=0)

nx.draw(G, with_labels=True, font_weight='bold')
print(tour)
