#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:20:27 2023

@author: shiming
"""

# %% Getting started with hypervolumes
# ---- Hypervolume interface and construction
from numpy import array
import pygmo as pg
from pygmo import hypervolume
print('hypervolume' in dir())

# Construct a DTLZ-2 problem with 3-dimensional fitness space and 10 dimensions
udp = pg.problem(pg.dtlz(prob_id=2, dim=10, fdim=3))
pop = pg.population(udp, 50)
hv = hypervolume(pop)

hv = hypervolume(array([[1, 0], [0.5, 0.5], [0, 1]]))
hv_from_list = hypervolume([[1, 0], [0.5, 0.5], [0, 1]])

# test maximization
hv = hypervolume(-array([[1, 0], [0.5, 0.5], [0, 1]]))
ref_point = array([1, 1])
print('hypervolume_maximization: ', hv.compute(ref_point))

# ---- Computing the hypervolume indicator and hypervolume contributions
hv = hypervolume([[1, 0], [0.5, 0.5], [0, 1], [1.5, 0.75]])
ref_point = [2, 2]
print('hypervolume: ', hv.compute(ref_point))
print('exclusive 1: ', hv.exclusive(1, ref_point))
print('exclusive 3: ', hv.exclusive(3, ref_point))
print('least_contributor: ', hv.least_contributor(ref_point))
print('greatest_contributor: ', hv.greatest_contributor(ref_point))
print('contributions: ', hv.contributions(ref_point))

# Instantiates a 4-objectives problem
prob = pg.problem(pg.dtlz(prob_id=4, dim=12, fdim=4))
pop = pg.population(prob, 84)
# Construct the hypervolume object
# and get the reference point off-setted by 10 in each objective
hv = pg.hypervolume(pop)
ref_point = hv.refpoint(offset=1)
print('hypervolume before: ', hv.compute(ref_point))
# Evolve the population some generations
algo = pg.algorithm(pg.moead(gen=2000))
pop = algo.evolve(pop)
# Compute the hypervolume indicator again.
# This time we expect a higher value as SMS-EMOA evolves the population
# by trying to maximize the hypervolume indicator.
hv = pg.hypervolume(pop)
print('hypervolume after: ', hv.compute(ref_point))

# %% Advanced hypervolume computation and analysis
# ---- Computing hypervolume using a specific algorithm
hv = pg.hypervolume([[1, 0, 1], [1, 1, 0], [-1, 2, 2]])
print('hypervolume_hvwfg: ', hv.compute([5, 5, 5], hv_algo=pg.hvwfg()))
# hv2d, hv3d, hvwfg, bf_fpras, bf_approx

# %% Approximating the hypervolume
# ---- bf_fpras: capable of approximating the hypervolume indicator
prob = pg.problem(pg.dtlz(prob_id=3, fdim=10, dim=11))
pop = pg.population(prob, 100)
fpras = pg.bf_fpras(eps=0.1, delta=0.1)
hv = pg.hypervolume(pop)
ref_point = hv.refpoint(offset=0.1)
print('hv_appr: ', hv.compute(ref_point, hv_algo=fpras))
# print('hv_exact: ', hv.compute(ref_point))

# ---- bf_approx: capable of approximating the least and the greatest contributor
# this one seems not working
# Problem with 30 objectives and 300 individuals
prob = pg.problem(pg.dtlz(prob_id=3, fdim=30, dim=35))
pop = pg.population(prob, size=300)
hv_algo = pg.bf_approx(eps=0.1, delta=0.1)
hv = pg.hypervolume(pop)
ref_point = hv.refpoint(offset=0.1)
print('least_contr_appr: ', hv.least_contributor(ref_point, hv_algo=hv_algo))
# print('least_contr_exact: ', hv.least_contributor(ref_point))
