#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:25:37 2023

@author: shiming
"""

# %% import
from copy import deepcopy
from operator import itemgetter
import itertools
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt


# %% local functions
# ---- formulate the OP problem
def formulate_OP(df):
    # remove all na columns
    df = df.dropna(axis='columns', how='all')
    # get distance matrix
    distance_matrix = df.loc['D'].to_numpy()
    # drop rows with index 'D'
    df = df.drop(labels='D')
    df = df.dropna(axis='columns', how='all')

    # seperate data from configs
    data = df.dropna(axis='rows', how='any').to_numpy()
    # generate a complete graph
    n = int(df.loc['N', 1])
    OP_formulation = nx.complete_graph(n)
    # enter graph attr
    OP_formulation.graph['n'] = n
    OP_formulation.graph['maximum_path_length'] = df.loc['U', 2]
    OP_formulation.graph['start'] = int(df.loc['B', 1])
    OP_formulation.graph['end'] = int(df.loc['E', 1])
    # enter node attr
    for i in OP_formulation.nodes:
        OP_formulation.nodes[i]['objectives'] = data[i, 2:]
    # enter edge attr
    for i in OP_formulation.nodes:
        for j in OP_formulation.nodes:
            if j > i:
                OP_formulation.edges[i, j]['distance'] = distance_matrix[i, j]
            elif j == i:
                OP_formulation.add_edge(j, i, distance=0.0)

    return OP_formulation


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
    if bee['solution'][0] != bee['solution'][-1]:
        # situation where start != end
        for i in bee['solution']:
            objectives += OP_formulation.nodes[i]['objectives']
    else:
        # situation where start == end
        for i in bee['solution'][:-1]:
            objectives += OP_formulation.nodes[i]['objectives']
    bee['objectives'] = objectives
    return bee


def pareto_filter(list_of_bees):
    """
    Take in list of bees, output the pareto_front_approximation, pareto_set_approximation

    Parameters
    ----------
    list_of_bees : list
        A list of bees.

    Returns
    -------
    pareto_front_approximation : list
        The objective vectors of the Pareto front approximation.
    pareto_set_approximation : list
        The decision vectors of the Pareto set approximation.

    """
    pareto_front_approximation = []
    pareto_set_approximation = []
    # get the set of bees corresponding to the Pareto approximation
    # init the first non-dominated front: F_0
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
            dominance_indicator = check_dominance(p, q)
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
    # collect the objective vectors
    for bee_idx in F0:
        pareto_front_approximation.append(list_of_bees[bee_idx]['objectives'])
    # collect the decision vectors
    for bee_idx in F0:
        pareto_set_approximation.append(list_of_bees[bee_idx]['solution'])
    return pareto_front_approximation, pareto_set_approximation


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
    num_best_next = min(OP_formulation.graph['n'], 60)

    # calculate ratio for each node and find the best next node
    n = len(OP_formulation)
    for i in OP_formulation.nodes:
        ratio_ij = np.zeros(n)
        for j in OP_formulation.neighbors(i):
            objective_sum = np.sum(OP_formulation.nodes[j]['objectives'])
            distance = OP_formulation.edges[(i, j)]['distance']
            if i != j:
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
                if delta_path_length > 1e-5:
                    desirable_inversions.append([j+1, i-1, delta_path_length])
    return desirable_inversions


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
    # shorten
    new_bee = shorten(new_bee, OP_formulation)
    # exchange
    new_bee = exchange(new_bee, OP_formulation)
    # get objectives
    new_bee = get_objectives(new_bee, OP_formulation)
    return new_bee


# ---- solution evaluation & selection
def check_dominance(bee, another_bee):
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
    if (bee['objectives'] >= another_bee['objectives']).all() and (bee['objectives'] > another_bee['objectives']).any():
        dominance_indicator = 1
    elif (bee['objectives'] <= another_bee['objectives']).all() and (bee['objectives'] < another_bee['objectives']).any():
        dominance_indicator = -1
    else:
        dominance_indicator = 0
    return dominance_indicator


def rank(list_of_bees):
    """
    Assign the non-dominated front number(rank) to the bees in the list

    Parameters
    ----------
    list_of_bees : list
        A list of bees without rank value.

    Returns
    -------
    list_of_fronts : list
        A list of non-dominated fronts.

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
            dominance_indicator = check_dominance(p, q)
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
    # init list_of_fronts
    list_of_fronts = []
    # loop through F
    for Fi in F[:-1]:
        # set front
        front = [list_of_bees[bee_idx] for bee_idx in Fi]
        list_of_fronts.append(front)
    return list_of_fronts


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
        if len(list_of_bees) > 2:
            # other bees' crowd increments by the crowd between its 2 neighbors
            # devided by the maximum crowd for this objective
            maximum_crowd = list_of_bees[-1]['objectives'][obj_idx] - \
                list_of_bees[0]['objectives'][obj_idx]
            if maximum_crowd == 0:
                maximum_crowd = np.inf
            for bee_idx in range(len(list_of_bees[1:-1])):
                # fix index
                bee_idx += 1
                # calculate crowd
                smaller_obj = list_of_bees[bee_idx-1]['objectives'][obj_idx]
                larger_obj = list_of_bees[bee_idx+1]['objectives'][obj_idx]
                crowd = (larger_obj-smaller_obj)/maximum_crowd
                list_of_bees[bee_idx]['crowd'] += crowd
    return list_of_bees


def multi_objectives_sort(list_of_bees):
    """
    Sort the list of bees by rank and then crowd

    Parameters
    ----------
    list_of_bees : list
        A list of bees.

    Returns
    -------
    sorted_list_of_bees : list
        A list of bees sorted by rank and then crowd.

    """
    sorted_list_of_bees = None
    # get list of fronts
    list_of_fronts = rank(list_of_bees)
    # loop through list_of_fronts
    for front_idx in range(len(list_of_fronts)):
        # sort the front
        list_of_fronts[front_idx] = crowd(list_of_fronts[front_idx])
        list_of_fronts[front_idx].sort(key=itemgetter('crowd'), reverse=True)
    # merge the fronts
    sorted_list_of_bees = list(itertools.chain.from_iterable(list_of_fronts))
    return sorted_list_of_bees


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


# %% MOABC formulation
def MOABC(num_bee, maximum_trial_num, maximum_itr, OP_formulation, label):
    # ---- Initialization Phase
    # generate best movements for each nodes
    get_best_movements(OP_formulation)
    # generate employed bees
    bees = []
    for i in range(num_bee):
        bees.append(create_bee(OP_formulation))
    # structs for logging
    best_bees = []
    for itr in tqdm(range(maximum_itr)):
        # ---- Employed Bees Phase
        for employed_bee_idx, employed_bee in enumerate(bees):
            # mutate: shorten, insert
            neighbor_bee = mutate_employed(employed_bee, OP_formulation)
            # replace current solution according to dominant relationship
            dominance_indicator = check_dominance(employed_bee, neighbor_bee)
            if dominance_indicator >= 0:
                # employed_bee is not dominated
                bees[employed_bee_idx]['trial_num'] += 1
            else:
                # neighbor_bee dominates employed_bee
                bees[employed_bee_idx] = neighbor_bee
                bees[employed_bee_idx]['trial_num'] = 0
        # ---- Onlooker Bees Phase
        # multi-objective sort
        bees = multi_objectives_sort(bees)
        # get probabiities
        selection_probability = get_probability(bees)
        for onlooker_bee_idx in range(num_bee):
            # sample the employed bees
            selected_employed_bee_idx = rng.choice(
                num_bee, p=selection_probability)
            onlooker_bee = bees[selected_employed_bee_idx]
            # mutate: shorten, exchange
            neighbor_bee = mutate_onlooker(onlooker_bee, OP_formulation)
            # replace current solution according to dominant relationship
            dominance_indicator = check_dominance(onlooker_bee, neighbor_bee)
            if dominance_indicator >= 0:
                # onlooker_bee is not dominated by neighbor_bee
                bees[selected_employed_bee_idx]['trial_num'] += 1
            else:
                # neighbor_bee dominate onlooker_bee
                bees[selected_employed_bee_idx] = neighbor_bee
                bees[selected_employed_bee_idx]['trial_num'] = 0
        # ---- Scout Bees Phase
        # reset employed bees if necessary
        for bee_idx, bee in enumerate(bees):
            if bee['trial_num'] > maximum_trial_num:
                bees[bee_idx] = create_bee(OP_formulation)
        # ---- Memorize the best solution achieved so far
        # log and save best obj for record
        best_bees.extend(bees)
    # get the Pareto approximations
    pareto_front_approximation, pareto_set_approximation = pareto_filter(
        best_bees)
    return pareto_front_approximation, pareto_set_approximation


# %% global seed
rng = np.random.default_rng(0)

# find the paths of all the txt files
txt_paths = list(
    Path("../../../dataset/moop/2 objectives/dmatrix").rglob("*_t*.[tT][xX][tT]"))

# %% problem formulation
num_bee = 10
maximum_trial_num = 10
maximum_itr = 20
runs_num = 10
for txt_path in txt_paths:
    print(txt_path.stem)
    # init figure
    plt.figure()
    plt.xlabel('obj_1')
    plt.ylabel('obj_2')
    plt.title('Pareto front approximations')
    for run in range(runs_num):
        df = pd.read_csv(txt_path, comment='/',
                         names=list(range(2144)), on_bad_lines='skip', index_col=0)
        OP_formulation = formulate_OP(df)
        pareto_front_approximation, pareto_set_approximation = MOABC(num_bee, maximum_trial_num, maximum_itr,
                                                                     OP_formulation, txt_path.stem)
        # write files
        pareto_front_approximation = pd.DataFrame(pareto_front_approximation)
        pareto_set_approximation = pd.DataFrame(pareto_set_approximation)
        pareto_front_approximation_csv = pareto_front_approximation.to_csv(
            index=False, header=False, sep=' ')
        pareto_set_approximation_csv = pareto_set_approximation.to_csv(
            index=False, header=False)
        with open('results/'+txt_path.stem+'_MOABC_front', 'a') as file:
            file.write(pareto_front_approximation_csv+'\n')
        with open('results/'+txt_path.stem+'_MOABC_set', 'a') as file:
            file.write(pareto_set_approximation_csv+'\n')
        # plot fronts
        if run < 5:
            pareto_front_approximation = pareto_front_approximation.sort_values(
                by=0)
            plt.step(
                pareto_front_approximation[0], pareto_front_approximation[1], 'o-')
    plt.savefig('results/'+txt_path.stem+'_MOABC_front.jpg')
    plt.close()


"""
comment:
    parse standard dataset from txt files, the dominance check now complies with
    strict dominance instead of weakly dominance, the output is a csv of Pareto
    approximations
"""
