#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:12:20 2023

@author: shiming
"""

# %% imports
from math import dist
from copy import deepcopy
from operator import itemgetter
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.impute import KNNImputer
from matplotlib import pyplot as plt

# %% global seed
rng = np.random.default_rng(0)

# %% problem formulation
# ---- load data
# load from xlsx
df = pd.read_excel('../../dataset/SHF_Uncertainty.xlsx')
# drop duplicate points by name
df = df.drop_duplicates(subset=['StraboSpot Dataset Download: '])
# turn dataframe into array
raw = df.to_numpy()
# take the last 5 col: ['Attachedness', 'Lithology', 'Geometry', 'lat', 'long']
raw_digit = raw[:, -5:].astype(float)
# remove all nan rows
data_miss = raw_digit[~np.isnan(raw_digit[:, :3]).all(axis=1)]

# ---- impute data
imputer = KNNImputer()

data_miss_scale = data_miss.copy()
data_miss_scale[:, :3] /= 5
data_miss_scale[:, -2:] *= 20
data_impute_scale = imputer.fit_transform(data_miss_scale)
data_impute = data_impute_scale.copy()
data_impute[:, :3] *= 5
data_impute[:, -2:] /= 20
data_impute[:, :3] = np.round(data_impute[:, :3])

# ---- formulate the OP problem
# generate a complete graph
n = data_impute.shape[0]
OP_formulation = nx.complete_graph(n)
# enter graph attr
OP_formulation.graph['maximum_path_length'] = 0.2
OP_formulation.graph['start'] = 0
OP_formulation.graph['end'] = 224
# enter node attr
for i in OP_formulation.nodes:
    OP_formulation.nodes[i]['objectives'] = data_impute[i, :3]
    OP_formulation.nodes[i]['lat'] = data_impute[i, 3]
    OP_formulation.nodes[i]['long'] = data_impute[i, 4]
# enter edge attr
for i in OP_formulation.nodes:
    for j in OP_formulation.nodes:
        if j > i:
            pos_i = [OP_formulation.nodes[i]['lat'],
                     OP_formulation.nodes[i]['long']]
            pos_j = [OP_formulation.nodes[j]['lat'],
                     OP_formulation.nodes[j]['long']]
            distance_ij = dist(pos_i, pos_j)
            OP_formulation.edges[i, j]['distance'] = distance_ij

# total_distance = 0
# for i, j, a in OP_formulation.edges(data=True):
#     print(a)
#     total_distance += a['distance']
# for i, a in OP_formulation.nodes(data=True):
#     print(a)

# %% local functions


# ---- util functions
def get_path_length(bee, OP_formulation):
    """
    Return the path length of a bee solution

    Parameters
    ----------
    bee : dict
        A bee instance.
    OP_formulation : Graph
        A networkx graph describing the OP problem, including nodes(vertices),
        scores, costs and best_movements.

    Returns
    -------
    path_length : float
        Path length of the bee solution.

    """
    # get the number of nodes in the solution
    n = len(bee['solution'])
    # init the path length
    path_length = 0
    for i in range(n-1):
        e = (bee['solution'][i], bee['solution'][i+1])
        path_length += OP_formulation.edges[e]['distance']
    return path_length


def get_objectives(bee, OP_formulation):
    """
    Calculate and update the objectives of a bee

    Parameters
    ----------
    bee : dict
        A bee instance.
    OP_formulation : Graph
        A networkx graph describing the OP problem, including nodes(vertices),
        scores, costs.

    Returns
    -------
    bee : dict
        A bee instance.

    """
    # calculate objectives
    objectives = np.zeros_like(OP_formulation.nodes[0]['objectives'])
    for i in bee['solution']:
        objectives += OP_formulation.nodes[i]['objectives']
    bee['objectives'] = objectives
    return bee


# ---- initialization phase
def get_best_movements(OP_formulation):
    """
    calculate the greedy best movements for each node

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including nodes(vertices),
        scores, costs.

    Returns
    -------
    OP_formulation : Graph
        The same networkx graph with additional attributes best_movements
        (A list of nodes) for each node.

    """
    # hyperparams
    num_best_next = 40

    # calculate ratio for each node and find the best next node
    n = len(OP_formulation)
    for i in OP_formulation.nodes:
        ratio_ij = np.zeros(n)
        for j in OP_formulation.neighbors(i):
            objective_sum = np.sum(OP_formulation.nodes[j]['objectives'])
            distance = OP_formulation.edges[(i, j)]['distance']
            if distance == 0:
                print(i, j)
            ratio_ij[j] = objective_sum/distance
        # find the index corresponding to the 10 largest ratio
        best_next = np.argpartition(ratio_ij, -num_best_next)[-num_best_next:]
        OP_formulation.nodes[i]['best_next'] = best_next
    return OP_formulation


def create_bee(OP_formulation):
    """
    Create a bee by a stochastic local greedy strategy

    choose the next location randomly among the best_next until the path length
    exceed the maximum_path_length

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including nodes(vertices),
        scores, costs and best_movements.

    Returns
    -------
    bee : dict
        A bee instance with keys: solution.

    """

    bee = None

    # bee: {solution: [0, ..., 224], rank: nan, crowd: nan}
    # graph attr shorthand
    start = OP_formulation.graph['start']
    end = OP_formulation.graph['end']
    maximum_path_length = OP_formulation.graph['maximum_path_length']
    # init at the starting node
    bee = {}
    bee['trial_num'] = 0
    bee['solution'] = [start]
    path_length = 0
    # while loop
    while True:
        # choose next node among the best_next of current node
        current_node = bee['solution'][-1]
        best_next = OP_formulation.nodes[current_node]['best_next']
        best_next_shuffled = rng.permuted(best_next)
        # check if the next node exists in the solution
        for next_node in best_next_shuffled:
            if next_node not in bee['solution'] and next_node != end:
                break
        # print if the whole best_next exists in the solution
        if next_node in bee['solution']:
            print(
                'create_bee: all of the best next are visited, try enlarging n in get_best_movements.')
        # append next node to the solution
        bee['solution'].append(next_node)
        # calculate path length
        path_length += OP_formulation.edges[(current_node,
                                             next_node)]['distance']
        # check if exceeding maximum path length
        if path_length+OP_formulation.edges[(next_node, end)]['distance'] > maximum_path_length:
            # if true, replace the last node with the end node
            bee['solution'][-1] = end
            break
    # get objectives
    bee = get_objectives(bee, OP_formulation)
    return bee


# ---- mutates
def shorten(old_bee, OP_formulation):
    """
    Create a new bee with better solution by inverting the subpath of the 
    solution of the old bee

    Parameters
    ----------
    old_bee : dict
        A bee instance.

    Returns
    -------
    new_bee : dict
        A bee instance.

    """
    new_bee = deepcopy(old_bee)
    # while true
    while True:
        # build the distance matrix where the solution is the slant
        rearranged_matrix = build_rearranged_matrix(new_bee, OP_formulation)
        # loop through the lower triangle to find desirable inversions
        desirable_inversions = find_desirable_inversions(rearranged_matrix)
        # break if the inversion set is empty
        if not desirable_inversions:
            break
        # pick the best one in each group
        desirable_inversions = sorted(desirable_inversions, key=itemgetter(2))
        best_inversion = desirable_inversions[-1][:2]
        # invert the subpath in the solution
        # the inversion index are inclusive, hence +1
        new_bee['solution'][best_inversion[0]:best_inversion[1]+1] = \
            new_bee['solution'][best_inversion[0]:best_inversion[1]+1][::-1]
        # # test
        # print(new_bee['solution'], best_inversion,
        #       get_path_length(new_bee, OP_formulation))
    return new_bee


def build_rearranged_matrix(bee, OP_formulation):
    """
    Given a bee and a OP_formulation, compose a rearranged_matrix such that the
    solution of the bee is the slant of the rearranged_matrix

    Parameters
    ----------
    bee : dict
        A bee instance.
    OP_formulation : Graph
        A networkx graph describing the OP problem, including nodes(vertices),
        scores, costs and best_movements.

    Returns
    -------
    rearranged_matrix : ndarray
        n*n array with the rearranged distance entries.

    """
    rearranged_matrix = None

    # get the length of the solution and init the matrix
    n = len(bee['solution'])
    rearranged_matrix = np.zeros((n, n))
    # loop through the solution
    for rearranged_i, i in enumerate(bee['solution']):
        # loop through the solution
        for rearranged_j, j in enumerate(bee['solution']):
            if rearranged_i > rearranged_j:
                rearranged_matrix[rearranged_i, rearranged_j] = \
                    OP_formulation.edges[(i, j)]['distance']
    return rearranged_matrix

# # test
# test_bee = {}
# test_bee['solution'] = [2, 4, 6, 8, 10, 12, 14, 16, 18]
# test_rearranged_matrix = build_rearranged_matrix(test_bee, OP_formulation)
# print(test_rearranged_matrix)


def find_desirable_inversions(rearranged_matrix):
    """
    Given a rearranged_matrix, find the desirable inversions that reduce the
    path length

    Parameters
    ----------
    rearranged_matrix : ndarray
        n*n array with the rearranged distance entries.

    Returns
    -------
    desirable_inversions : list
        [..., [rearranged_i, rearranged_j, delta_path_length], ...].

    """
    desirable_inversions = []

    # get the size of the rearranged_matrix and init the list
    n = rearranged_matrix.shape[0]
    # loop i from 3 to n-1
    for i in range(3, n):
        # loop j from 3 to n-1
        for j in range(0, n-3):
            # check whether desirable if j > i+2
            if i > j+2:
                delta_path_length = rearranged_matrix[i, i-1] +\
                    rearranged_matrix[j+1, j]-rearranged_matrix[i, j+1] -\
                    rearranged_matrix[i-1, j]
                if delta_path_length > 0:
                    desirable_inversions.append([j+1, i-1, delta_path_length])
    return desirable_inversions

# # test
# test_desirable_inversions = find_desirable_inversions(test_rearranged_matrix)
# print(test_desirable_inversions)

# test_desirable_inversions = sorted(
#     test_desirable_inversions, key=itemgetter(2))
# test_best_inversion = test_desirable_inversions[-1][:2]
# print(test_best_inversion)

# test_bee['solution'][test_best_inversion[0]:test_best_inversion[1]+1] = \
#     test_bee['solution'][test_best_inversion[0]:test_best_inversion[1]+1][::-1]
# print(test_bee['solution'])

# test_new_bee = shorten(test_bee, OP_formulation)
# print(test_new_bee['solution'])


def insert(old_bee, OP_formulation):
    """
    Create a new bee by inserting new points to the solution of the old bee

    Parameters
    ----------
    old_bee : dict
        A bee instance.

    Returns
    -------
    new_bee : dict
        A bee instance.

    """
    # init new bee
    new_bee = deepcopy(old_bee)
    # while true
    while True:
        # init insertion_candidates
        insertion_candidates = []
        # get current path length
        new_path_length = get_path_length(new_bee, OP_formulation)
        # loop through node 0 ~ n-2
        for idx, current_node in enumerate(new_bee['solution'][:-1]):
            # loop through the best next nodes
            for insert_node in OP_formulation.nodes[current_node]['best_next']:
                if insert_node not in new_bee['solution']:
                    # get path length after insertion
                    e_new = (current_node, new_bee['solution'][idx+1])
                    e_trial_1 = (current_node, insert_node)
                    e_trial_2 = (insert_node, new_bee['solution'][idx+1])
                    distance_new = OP_formulation.edges[e_new]['distance']
                    distance_trial = OP_formulation.edges[e_trial_1]['distance'] + \
                        OP_formulation.edges[e_trial_2]['distance']
                    trial_path_length = new_path_length-distance_new+distance_trial
                    # if the new path length is less than maximum
                    if trial_path_length < OP_formulation.graph['maximum_path_length']:
                        # create trial bee
                        trial_bee = deepcopy(new_bee)
                        trial_bee['solution'].insert(idx+1, insert_node)
                        # append to the insertion candidates
                        insertion_candidates.append(trial_bee)
        # break if the insertion candidates is empty
        if not insertion_candidates:
            break
        # randomly choose one to insert
        new_bee = rng.choice(insertion_candidates)
        # # test
        # path_length = get_path_length(new_bee, OP_formulation)
        # print(path_length)
    return new_bee


# # test
# get_best_movements(OP_formulation)
# test_bee = create_bee(OP_formulation)
# test_new_bee = insert(test_bee, OP_formulation)


def exchange(old_bee, OP_formulation):
    """
    Create a new bee by replacing points of the solution of the old bee

    Parameters
    ----------
    old_bee : dict
        A bee instance.

    Returns
    -------
    new_bee : dict
        A bee instance.

    """
    # init new bee
    new_bee = deepcopy(old_bee)
    # loop through the old solution
    for idx, current_node in enumerate(new_bee['solution'][1:-1]):
        # init exchange candidates
        exchange_candidates = []
        # fix idx
        idx += 1
        # get current path length
        new_path_length = get_path_length(new_bee, OP_formulation)
        # loop through the best next nodes
        for exchange_node in OP_formulation.nodes[current_node]['best_next']:
            if exchange_node not in new_bee['solution']:
                # check path length
                e_new_1 = (new_bee['solution'][idx-1], current_node)
                e_new_2 = (current_node, new_bee['solution'][idx+1])
                e_trial_1 = (new_bee['solution'][idx-1], exchange_node)
                e_trial_2 = (exchange_node, new_bee['solution'][idx+1])
                distance_new = OP_formulation.edges[e_new_1]['distance'] + \
                    OP_formulation.edges[e_new_2]['distance']
                distance_trial = OP_formulation.edges[e_trial_1]['distance'] + \
                    OP_formulation.edges[e_trial_2]['distance']
                trial_path_length = new_path_length-distance_new+distance_trial
                # if path length < maximum path length
                if trial_path_length < OP_formulation.graph['maximum_path_length']:
                    # create trial bee
                    trial_bee = deepcopy(new_bee)
                    trial_bee['solution'][idx] = exchange_node
                    # append to exchange candidate
                    exchange_candidates.append(trial_bee)
        # randomly choose one if exchange_candidates is not empty
        if exchange_candidates:
            new_bee = rng.choice(exchange_candidates)
        # # test
        # path_length = get_path_length(new_bee, OP_formulation)
        # print(path_length)
    return new_bee


# # test
# get_best_movements(OP_formulation)
# test_bee = create_bee(OP_formulation)
# test_bee = insert(test_bee, OP_formulation)
# print('exchange')
# test_bee = exchange(test_bee, OP_formulation)


def mutate_employed(old_bee, OP_formulation):
    """
    Create a new bee by applying shorten, insert, exchange and check 
    dominance on the old bee

    Parameters
    ----------
    old_bee : dict
        A bee instance.

    Returns
    -------
    new_bee : dict
        A bee instance.

    """
    new_bee = deepcopy(old_bee)
    # shorten
    new_bee = shorten(new_bee, OP_formulation)
    # insert
    new_bee = insert(new_bee, OP_formulation)
    # get objectives
    new_bee = get_objectives(new_bee, OP_formulation)
    return new_bee


# # test
# get_best_movements(OP_formulation)
# test_bee = create_bee(OP_formulation)
# test_new_bee = mutate_employed(test_bee, OP_formulation)


def mutate_onlooker(old_bee, OP_formulation):
    """
    Create a new bee by applying shorten, insert, exchange and check 
    dominance on the old bee

    Parameters
    ----------
    old_bee : dict
        A bee instance.

    Returns
    -------
    new_bee : dict
        A bee instance.

    """
    new_bee = deepcopy(old_bee)
    # exchange
    new_bee = exchange(new_bee, OP_formulation)
    # shorten
    new_bee = shorten(new_bee, OP_formulation)
    # get objectives
    new_bee = get_objectives(new_bee, OP_formulation)
    return new_bee


# # test
# get_best_movements(OP_formulation)
# test_bee = create_bee(OP_formulation)
# test_new_bee = mutate_onlooker(test_bee, OP_formulation)


# ---- solution evaluation & selection
def check_dominance(bee, another_bee, OP_formulation):
    """
    Compare the dominance relationship of 2 bees, return the dominant bee if
    one bee dominate another, otherwise, return None

    Parameters
    ----------
    bee : dict
        A bee instance.
    another_bee : dict
        A bee instance.

    Returns
    -------
    dominance_indicator : int
        1 if bee dominates another_bee, -1 if another_bee dominates bee, 0 otherwise.

    """
    dominance_indicator = None
    # set dominant_bee
    if (bee['objectives'] > another_bee['objectives']).all():
        dominance_indicator = 1
    elif (bee['objectives'] < another_bee['objectives']).all():
        dominance_indicator = -1
    else:
        dominance_indicator = 0
    return dominance_indicator


# # test
# get_best_movements(OP_formulation)
# test_bee = create_bee(OP_formulation)
# test_another_bee = mutate_employed(test_bee, OP_formulation)
# test_dominant_bee = check_dominance(test_bee, test_another_bee, OP_formulation)


def rank(list_of_bees):
    """
    Assign the non-dominated front number(rank) to the bees in the list

    Parameters
    ----------
    list_of_bees : list
        A list of bees without rank value.

    Returns
    -------
    list_of_bees : list
        A list of bees with rank value.

    """
    # init the first non-dominated front: F_0
    F = []
    F0 = set()
    # for each p in P
    for p_idx, p in enumerate(list_of_bees):
        # the set of solutions dominated by p: S_p = empty set
        list_of_bees[p_idx]['S'] = set()
        # the domination counter of p: n_p = 0
        list_of_bees[p_idx]['n'] = 0
        # for each q in P
        for q_idx, q in enumerate(list_of_bees):
            # check dominance
            dominance_indicator = check_dominance(p, q, OP_formulation)
            if dominance_indicator == 1:
                # if p dominates q then add q to S_p
                list_of_bees[p_idx]['S'].add(q_idx)
            elif dominance_indicator == -1:
                # else if q dominates p then increment n_p
                list_of_bees[p_idx]['n'] += 1
        # if n_p == 0
        if list_of_bees[p_idx]['n'] == 0:
            # p_rank = 0
            list_of_bees[p_idx]['rank'] = 0
            # add p to F_0
            F0.add(p_idx)
    # append F0 to F
    F.append(F0)
    # i = 0
    i = 0
    # while F_i isn't empty
    while F[i]:
        # Q = empty set
        Q = set()
        # for each p in F_i
        for p_idx in F[i]:
            # for each q in S_p
            for q_idx in list_of_bees[p_idx]['S']:
                # n_q = n_q-1
                list_of_bees[q_idx]['n'] -= 1
                # if n_q == 0
                if list_of_bees[q_idx]['n'] == 0:
                    # q_rank = i+1
                    list_of_bees[q_idx]['rank'] = i+1
                    # add q to Q
                    Q.add(q_idx)
        # i = i+1
        i += 1
        # F_i = Q
        F.append(Q)
    return list_of_bees


# # test
# get_best_movements(OP_formulation)
# list_of_bees = []
# for i in range(100):
#     list_of_bees.append(create_bee(OP_formulation))
# list_of_bees = rank(list_of_bees)


def crowd(list_of_bees):
    """
    Assign the crowding distance to the bees in the list

    Parameters
    ----------
    list_of_bees : list
        A list of bees without crowd value.

    Returns
    -------
    list_of_bees : list
        A list of bees with crowd value.

    """
    # for each bee, set its crowd to 0
    for bee_idx in range(len(list_of_bees)):
        list_of_bees[bee_idx]['crowd'] = 0
    # for each objective
    for obj_idx in range(len(list_of_bees[0]['objectives'])):
        # sort the bees according to this objectove
        list_of_bees.sort(key=lambda x: x['objectives'][obj_idx])
        # the first and the last bee has infinite crowd
        list_of_bees[0]['crowd'] = np.inf
        list_of_bees[-1]['crowd'] = np.inf
        # other bees' crowd increments by the crowd between its 2 neighbors
        # devided by the maximum crowd for this objective
        maximum_crowd = list_of_bees[-1]['objectives'][obj_idx] - \
            list_of_bees[0]['objectives'][obj_idx]
        for bee_idx in range(len(list_of_bees[1:-1])):
            # fix index
            bee_idx += 1
            # calculate crowd
            smaller_obj = list_of_bees[bee_idx-1]['objectives'][obj_idx]
            larger_obj = list_of_bees[bee_idx+1]['objectives'][obj_idx]
            crowd = (larger_obj-smaller_obj)/maximum_crowd
            list_of_bees[bee_idx]['crowd'] += crowd
    return list_of_bees


# # test
# get_best_movements(OP_formulation)
# list_of_bees = []
# for i in range(100):
#     list_of_bees.append(create_bee(OP_formulation))
# list_of_bees = crowd(list_of_bees)
# list_of_bees = rank(list_of_bees)


def get_probability(list_of_bees):
    """
    Calculate the probability distribution across the bees in the list 
    according to their rank and crowd value

    Parameters
    ----------
    list_of_bees : list
        A list of bees with rank and crowd value.

    Returns
    -------
    selection_probability : ndarray
        The probability of each bee being selected.

    """
    selection_probability = None
    # init selection_probability
    selection_probability = np.empty(len(list_of_bees))
    # calculate MOfitness
    for bee_idx, bee in enumerate(list_of_bees):
        selection_probability[bee_idx] = 1/(2**bee['rank']+1/(1+bee['crowd']))
    # normalize selection_probability
    selection_probability /= np.sum(selection_probability)
    return selection_probability


# # test
# get_best_movements(OP_formulation)
# list_of_bees = []
# for i in range(100):
#     list_of_bees.append(create_bee(OP_formulation))
# list_of_bees = crowd(list_of_bees)
# list_of_bees = rank(list_of_bees)
# selection_probability = get_probability(list_of_bees)


# %% MOABC formulation
# ---- constants
num_bee = 100
maximum_trial_num = 10
maximum_itr = 100
# ---- Initialization Phase
# generate best movements for each nodes
get_best_movements(OP_formulation)
# generate employed bees
employed_bees = []
for i in range(num_bee):
    employed_bees.append(create_bee(OP_formulation))
# generate onlooker bees
onlooker_bees = []
for i in range(num_bee):
    onlooker_bees.append(create_bee(OP_formulation))
# structs for logging
best_bees = []
best_objectives = []
# ---- Employed Bees Phase
for employed_bee_idx, employed_bee in enumerate(tqdm(employed_bees)):
    # mutate: shorten, insert
    neighbor_bee = mutate_employed(employed_bee, OP_formulation)
    # replace current solution according to dominant relationship
    dominance_indicator = check_dominance(
        employed_bee, neighbor_bee, OP_formulation)
    if dominance_indicator >= 0:
        # employed_bee is not dominated
        employed_bees[employed_bee_idx]['trial_num'] += 1
    else:
        # neighbor_bee dominates employed_bee
        employed_bees[employed_bee_idx] = neighbor_bee
# ---- Onlooker Bees Phase
# calculate rank
employed_bees = rank(employed_bees)
# calculate crowd
employed_bees = crowd(employed_bees)
# get probabiities
selection_probability = get_probability(employed_bees)
for onlooker_bee_idx, onlooker_bee in enumerate(tqdm(onlooker_bees)):
    # sample the employed bees
    selected_employed_bee = rng.choice(employed_bees, p=selection_probability)
    onlooker_bee['solution'] = selected_employed_bee['solution']
    # mutate: shorten, exchange
    neighbor_bee = mutate_onlooker(onlooker_bee, OP_formulation)
    # replace current solution according to dominant relationship
    dominance_indicator = check_dominance(
        onlooker_bee, neighbor_bee, OP_formulation)
    if dominance_indicator > 0:
        # onlooker_bee dominate neighbor_bee
        onlooker_bees[onlooker_bee_idx]['trial_num'] += 1
    else:
        # neighbor_bee is not dominated by onlooker_bee
        onlooker_bees[onlooker_bee_idx] = neighbor_bee
# ---- Scout Bees Phase
# reset employed bees if necessary
list_of_bees = employed_bees+onlooker_bees
# ---- Memorize the best solution achieved so far
# calculate rank
list_of_bees = rank(list_of_bees)
# calculate crowd
list_of_bees = crowd(list_of_bees)
# save the best population as employed bees
# get probabiities
selection_probability = get_probability(list_of_bees)
# get the best num_bee population
new_employed_bees_idx = np.argpartition(
    selection_probability, -num_bee)[-num_bee:]
employed_bees = list(itemgetter(*new_employed_bees_idx)(list_of_bees))
# print and save best obj for record
best_bee = list_of_bees[np.argmax(selection_probability)]
best_bees.append(best_bee)
best_objectives.append(best_bee['objectives'])
