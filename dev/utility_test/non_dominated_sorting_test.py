#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:51:46 2023

@author: shiming
"""

import pygmo as pg
import numpy as np

rng = np.random.default_rng(1)
points = rng.random((10, 3))

ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points)
print('the non dominated fronts: ', ndf)
print('the domination list: ', dl)
print('the domination count: ', dc)
print('the non domination ranks: ', ndr)

nadir = pg.nadir(points)
print('the nadir point: ', nadir)

ideal = pg.ideal(points)
print('the ideal point: ', ideal)

pareto_dominance = pg.pareto_dominance(points[0], points[9])
print('pareto_dominance: ', pareto_dominance)

# # only work for 2 objectives
# ndf = pg.non_dominated_front_2d(rng.random((10, 2)))
# print('the non dominated fronts: ', ndf)

crowding_distance = pg.crowding_distance(points[ndf[0]])
print('crowding_distance: ', crowding_distance)

sort_population_mo = pg.sort_population_mo(points)
print('sort_population_mo: ', sort_population_mo)
