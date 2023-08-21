#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:25:37 2023

@author: shiming
"""

# %% import
from pygmo import hypervolume
from heapq import heappush, heappop
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
import elkai
import itertools
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt


# %% local class
class PriorityQueue:
    def __init__(self):
        self.d = {}
        self.hq = []
        self.counter = itertools.count()

    def set_item(self, key, priority):
        if key in self.d:
            self.remove_item(key)
        count = next(self.counter)
        item = [priority, count, key]
        self.d[key] = item
        heappush(self.hq, item)

    def remove_item(self, key):
        item = self.d.pop(key)
        item[-1] = '<removed-item>'

    def get_item(self):
        while self.hq:
            priority, count, key = heappop(self.hq)
            if key != '<removed-item>':
                del self.d[key]
                return key

    def peek_smallest(self):
        while True:
            priority, count, key = self.hq[0]
            if key == '<removed-item>':
                heappop(self.hq)
            else:
                return key, priority


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


# ---- select(pareto_set_approximation)
def logsumexp(x):
    """
    Standard log sum exp trick to avoid over/underflow.
    The resulting normalized vector is given by np.exp(x - logsumexp(x)).

    Parameters
    ----------
    x : ndarray
        The vector to take exponential.

    Returns
    -------
    float
        The logsumexp term to be subtracted from x.

    """
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


def select(pareto_set_approximation):
    """
    Select 2 parents using roulette wheel.
    The probability for each candidate is proportional to the exponential of 
    its hypervolume contribution.

    Parameters
    ----------
    pareto_set_approximation : dict
        A dict of non-dominated solutions, with the key-value pair 
        of permutation-solution.

    Returns
    -------
    parents : list
        2 parents chosen by roulette wheel.

    """
    # write the objectives as a ndarray
    objectives_of_all_solutions = []
    for solution in pareto_set_approximation.values():
        objectives_of_all_solutions.append(solution['objectives'])
    objectives_of_all_solutions = np.array(objectives_of_all_solutions)
    # normalize the objectives
    scaler = MinMaxScaler()
    objectives_of_all_solutions = scaler.fit_transform(
        objectives_of_all_solutions)
    # deal with situation with all 0
    if not np.any(objectives_of_all_solutions):
        objectives_of_all_solutions = np.ones_like(objectives_of_all_solutions)
    # get the unique objectives and the counts
    unique, unique_inverse, unique_counts = np.unique(
        objectives_of_all_solutions, return_inverse=True, return_counts=True, axis=0)
    # calculate the hypervolume contribution
    # NOTICE! the hypervolume api assumes minimization, hence take the minus
    hv_unique = hypervolume(-unique)
    ref_point = np.zeros_like(unique[0, :])+0.1
    hv_contributions_unique = hv_unique.contributions(ref_point)
    # reconstruct the hypervolume contribution with counts considered
    hv_contributions_unique = hv_contributions_unique / unique_counts
    hv = hv_contributions_unique[unique_inverse]
    # take the exponential
    # NOTICE! with the log sum exp trick, hv_exp sums up to 1
    hv_exp = np.exp(hv-logsumexp(hv))
    # get the parents
    parents_indices = rng.choice(
        len(pareto_set_approximation), 2, p=hv_exp)
    parents = [list(pareto_set_approximation.values())[parents_index]
               for parents_index in parents_indices]
    return parents


# ---- visited_vertices_from_solution(OP_formulation, solution)
def visited_vertices_from_solution(OP_formulation, solution):
    """
    Get the list of visited vertices from a solution.

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including vertices, scores, 
        costs.
    solution : dict
        The solution from which to get the visited vertices.

    Returns
    -------
    visited_vertices : list
        A list of visited vertices corresponding to the solution.

    """
    # initialize the list
    visited_vertices = [OP_formulation.graph['start']]
    # take out the permutation for prettier look
    permutation = solution['permutation']
    # iterate through the permutation
    while True:
        visited_vertices.append(permutation[visited_vertices[-1]])
        if visited_vertices[-1] == OP_formulation.graph['end']:
            break
    # cast into ndarray
    visited_vertices = np.array(visited_vertices)
    return visited_vertices


# ---- crossover(parents)
def crossover(OP_formulation, parents):
    """
    Create a child given a pair of parents by taking the intersection of the
    visited vertices of the parents.

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including vertices, scores, 
        costs.
    parents : list
        A list containing 2 parents.

    Returns
    -------
    child : dict
        A child solution generted by the crossover process.
    weight : ndarray
        A point on a standard (K-1)-simplex proportional to the child objectives.

    """
    # unpack the parents
    dad, mum = parents
    # return dad and a random search if dad == mum
    if (dad['objectives'] == mum['objectives']).all():
        if (dad['permutation'] == mum['permutation']).all():
            weight = rng.random(dad['objectives'].shape)
            weight = weight / np.sum(weight)
            return dad, weight
    # get the lists of visited vertices of the parents
    visited_vertices_dad = visited_vertices_from_solution(OP_formulation, dad)
    visited_vertices_mum = visited_vertices_from_solution(OP_formulation, mum)
    # get the intersected visited vertices
    visited_vertices_mum = set(visited_vertices_mum)
    visited_vertices_child = [
        vertex for vertex in visited_vertices_dad if vertex in visited_vertices_mum]
    # get the child
    child = solution_from_visited_list(OP_formulation, visited_vertices_child)
    # get the weight
    weight = dad['objectives']+mum['objectives']
    weight = weight/np.sum(weight)
    return child, weight


# ---- pareto_update(pareto_set_appr_0, pareto_set_appr_1): pareto_set_appr
def pareto_update(old_pareto_set_approximation, solution):
    """
    add the solution to the pareto_set_appr if it is Pareto efficient
    remove any solution dominated by the solution
    indicate Pareto efficiency

    NOTICE! the old_pareto_set_appr must be Pareto efficient, or unexpected 
    behavior may occur!

    Parameters
    ----------
    old_pareto_set_approximation : dict
        A dict of non-dominated solutions, with the key-value pair 
        of permutation-solution.
    solution : dict
        The solution to be examined.

    Returns
    -------
    new_pareto_set_appr : dict
        A dict of non-dominated solutions, with the key-value pair 
        of permutation-solution.
    updated : bool
        True if the solution is efficient, and vice versa.

    """
    # check repetition
    if tuple(solution['permutation']) in old_pareto_set_approximation:
        # if another_solution dominate solution, set False updated
        updated = False
        # break
        return old_pareto_set_approximation, updated

    # prepare for dominance comparision
    updated = True
    new_pareto_set_approximation = deepcopy(old_pareto_set_approximation)
    objectives = np.array(solution['objectives'])

    # iterate through the pareto_set_appr
    for key, another_solution in old_pareto_set_approximation.items():
        another_objectives = np.array(another_solution['objectives'])

        if (objectives >= another_objectives).all() and (objectives > another_objectives).any():
            # if solution dominate another_solution, del another_solution
            del new_pareto_set_approximation[key]

        if (objectives <= another_objectives).all() and (objectives < another_objectives).any():
            # if another_solution dominate solution, set False updated
            updated = False
            # break
            break

    # if updated is True, add solution to new_pareto_set_appr
    if updated == True:
        new_pareto_set_approximation[tuple(
            solution['permutation'])] = deepcopy(solution)

    return new_pareto_set_approximation, updated


# ---- shorten(OP_formulation, old_solution): new_solution
def shorten(OP_formulation, old_solution):
    """
    Improve the solution by solving a TSP on the visited vertices, thereby 
    reducing the route length.
    The TSP is solved by LKH3.

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including vertices, scores, 
        costs.
    old_solution : dict
        The input solution before shortening.

    Returns
    -------
    new_solution : dict
        The output solution after shortening.

    """
    # get the visited vertices from the solution
    visited_vertices = visited_vertices_from_solution(
        OP_formulation, old_solution)
    # early stop if there is no need for shortening
    if len(visited_vertices) <= 4:
        return old_solution
    if OP_formulation.graph['start'] == OP_formulation.graph['end']:
        # if the problem requires a tour, the distance matrix comtains all the
        # vertices except for the last one
        node_list = visited_vertices[:-1]
        distance_matrix = nx.adjacency_matrix(
            OP_formulation, nodelist=node_list, weight='distance').todense()
    else:
        # if the problem requires a route, the distance matrix comtains all the
        # vertices
        distance_matrix = nx.adjacency_matrix(
            OP_formulation, nodelist=visited_vertices, weight='distance').todense()
        # add the dummy vertex that connect to the start and end vertices
        sum_distance = np.sum(distance_matrix)
        distance_matrix = np.pad(
            distance_matrix, ((1, 0), (1, 0)), 'constant', constant_values=sum_distance)
        distance_matrix[0, 0] = 0
        distance_matrix[1, 0] = 0
        distance_matrix[0, 1] = 0
        distance_matrix[-1, 0] = 0
        distance_matrix[0, -1] = 0
    # throw the distance matrix into LKH
    tsp = elkai.DistanceMatrix(distance_matrix)
    tsp_solution = tsp.solve_tsp(runs=1)
    # get the new visited vertices
    if OP_formulation.graph['start'] == OP_formulation.graph['end']:
        # perform the swaps
        new_visited_vertices = visited_vertices[tsp_solution]
    else:
        # reverse the route is it is not in the correct direction
        # subtract the tsp_solution by 1 to get the index without dummy vertex
        if tsp_solution[1] != 1:
            tsp_solution = np.flip(tsp_solution)-1
        else:
            tsp_solution = np.array(tsp_solution)-1
        new_visited_vertices = visited_vertices[tsp_solution[1:-1]]
    # create the new solution
    new_solution = solution_from_visited_list(
        OP_formulation, new_visited_vertices)
    return new_solution


# ---- drop(OP_formulation, old_solution, dropped_proportion): new_solution
def drop(OP_formulation, old_solution, dropped_proportion, weight):
    """
    Perform the drop operator on the old solution.
    It drop dropped_proportion of all the vertices.

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including vertices, scores, 
    old_solution : dict
        The solution to be modified.
    dropped_proportion : float
        The percentage of vertices to be dropped.
    weight : ndarray
        A point on a standard (K-1)-simplex guiding the search.

    Returns
    -------
    new_solution : dict
        The solution after modification.

    """
    # unpack the solution
    permutation = old_solution['permutation']
    # get the total number of vertices
    total_num_vertices = np.count_nonzero(
        permutation-np.arange(len(permutation)))
    # get the number of vertices to be dropped
    drop_num_vertices = int((total_num_vertices-1)*dropped_proportion)
    # if the number of vertices to be dropped is 0, return the old solution
    if drop_num_vertices == 0:
        return old_solution
    new_solution = deepcopy(old_solution)
    # create the min heap with the drop_tuple (prev, curr, next) as key
    # initialize the heapdict
    pq = PriorityQueue()
    # initialize the iteration
    prev_vertex = OP_formulation.graph['start']
    curr_vertex = permutation[prev_vertex]
    next_vertex = permutation[curr_vertex]
    # iterate through the permutation
    while curr_vertex != OP_formulation.graph['end']:
        drop_tuple = (prev_vertex, curr_vertex, next_vertex)
        # compute the benefit
        benefit = weight@OP_formulation.nodes[curr_vertex]['objectives']
        # compute the cost
        cost = 0.
        cost += OP_formulation.edges[prev_vertex, curr_vertex]['distance']
        cost += OP_formulation.edges[curr_vertex, next_vertex]['distance']
        cost -= OP_formulation.edges[prev_vertex, next_vertex]['distance']
        # compute the bcr
        bcr = benefit/cost
        # add the drop tuple to the heapdict
        pq.set_item(drop_tuple, bcr)
        # move to the next vertex
        prev_vertex = curr_vertex
        curr_vertex = next_vertex
        next_vertex = permutation[curr_vertex]
    # for i in range(number of vertices to be dropped)
    for i in range(drop_num_vertices):
        # pop the smallest one
        drop_tuple = pq.get_item()
        # drop_vertex
        new_solution = drop_vertex(OP_formulation, new_solution, drop_tuple)
        # get the new permutation
        permutation = new_solution['permutation']
        # adjust the priority queue
        (prev_vertex, curr_vertex, next_vertex) = drop_tuple
        if prev_vertex != OP_formulation.graph['start'] and next_vertex != OP_formulation.graph['end']:
            # remove the affected tuples
            prev_prev_vertex = np.where(permutation == prev_vertex)[0][0]
            affected_tuple = (prev_prev_vertex, prev_vertex, curr_vertex)
            pq.remove_item(affected_tuple)
            next_next_vertex = permutation[next_vertex]
            affected_tuple = (curr_vertex, next_vertex, next_next_vertex)
            pq.remove_item(affected_tuple)
            # push in new tuple
            new_tuple = (prev_prev_vertex, prev_vertex, next_vertex)
            benefit = weight@OP_formulation.nodes[prev_vertex]['objectives']
            cost = 0.
            cost += OP_formulation.edges[prev_prev_vertex,
                                         prev_vertex]['distance']
            cost += OP_formulation.edges[prev_vertex, next_vertex]['distance']
            cost -= OP_formulation.edges[prev_prev_vertex,
                                         next_vertex]['distance']
            bcr = benefit/cost
            pq.set_item(new_tuple, bcr)
            new_tuple = (prev_vertex, next_vertex, next_next_vertex)
            benefit = weight@OP_formulation.nodes[next_vertex]['objectives']
            cost = 0.
            cost += OP_formulation.edges[prev_vertex, next_vertex]['distance']
            cost += OP_formulation.edges[next_vertex,
                                         next_next_vertex]['distance']
            cost -= OP_formulation.edges[prev_vertex,
                                         next_next_vertex]['distance']
            bcr = benefit/cost
            pq.set_item(new_tuple, bcr)
        elif prev_vertex == OP_formulation.graph['start']:
            # remove the affected tuples
            next_next_vertex = permutation[next_vertex]
            affected_tuple = (curr_vertex, next_vertex, next_next_vertex)
            pq.remove_item(affected_tuple)
            # push in new tuple
            new_tuple = (prev_vertex, next_vertex, next_next_vertex)
            benefit = weight@OP_formulation.nodes[next_vertex]['objectives']
            cost = 0.
            cost += OP_formulation.edges[prev_vertex, next_vertex]['distance']
            cost += OP_formulation.edges[next_vertex,
                                         next_next_vertex]['distance']
            cost -= OP_formulation.edges[prev_vertex,
                                         next_next_vertex]['distance']
            bcr = benefit/cost
            pq.set_item(new_tuple, bcr)
        else:
            # remove the affected tuples
            prev_prev_vertex = np.where(permutation == prev_vertex)[0][0]
            affected_tuple = (prev_prev_vertex, prev_vertex, curr_vertex)
            pq.remove_item(affected_tuple)
            # push in new tuple
            new_tuple = (prev_prev_vertex, prev_vertex, next_vertex)
            benefit = weight@OP_formulation.nodes[prev_vertex]['objectives']
            cost = 0.
            cost += OP_formulation.edges[prev_prev_vertex,
                                         prev_vertex]['distance']
            cost += OP_formulation.edges[prev_vertex, next_vertex]['distance']
            cost -= OP_formulation.edges[prev_prev_vertex,
                                         next_vertex]['distance']
            bcr = benefit/cost
            pq.set_item(new_tuple, bcr)
    return new_solution


# ---- add(OP_formulation, old_solution): new_solution
def add(OP_formulation, old_solution, weight):
    """
    Add vertices to the solution until it hits the route length constraint.

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including vertices, scores, 
    old_solution : dict
        The solution to be modified.
    weight : ndarray
        A point on a standard (K-1)-simplex guiding the search.

    Returns
    -------
    new_solution : dict
        The solution after modification.

    """
    # unpack the solution
    permutation = old_solution['permutation']
    # create a list for the priority queues
    pq_list = []
    # iterate through the permutation with enumerate
    for curr_vertex, next_vertex in enumerate(permutation):
        # if the vertex is unvisited
        if curr_vertex == next_vertex and curr_vertex != OP_formulation.graph['start']:
            # create the priority queue
            pq = PriorityQueue()
            # compute the benefit
            benefit = weight@OP_formulation.nodes[curr_vertex]['objectives']
            # iterate through the visited vertices
            prev_vertex = OP_formulation.graph['start']
            next_vertex = permutation[prev_vertex]
            while True:
                # create add_tuple
                add_tuple = (prev_vertex, curr_vertex, next_vertex)
                # compute the cost
                cost = 0.
                cost += OP_formulation.edges[prev_vertex,
                                             curr_vertex]['distance']
                cost += OP_formulation.edges[curr_vertex,
                                             next_vertex]['distance']
                cost -= OP_formulation.edges[prev_vertex,
                                             next_vertex]['distance']
                # compute the negative bcr
                # NOTICE! It is negative because the pq is a min heap
                negative_bcr = -benefit/cost
                # add the add tuple to the heapdict
                pq.set_item(add_tuple, negative_bcr)
                # move on to the next vertex
                prev_vertex = next_vertex
                next_vertex = permutation[prev_vertex]
                # break if the end is reached
                if prev_vertex == OP_formulation.graph['end']:
                    break
            # add the priority queue to the list
            pq_list.append((pq, curr_vertex, benefit))
    # copy the old solution into the new solution
    new_solution = deepcopy(old_solution)
    # while true
    while pq_list:
        # sort the list of priority queues by the peeked negative bcr
        pq_list.sort(
            key=lambda pq_list_item: pq_list_item[0].peek_smallest()[1])
        # set the infeasible flag to true
        infeasible = True
        # loop through the list of priority queue
        for pq_idx, (pq, _, _) in enumerate(pq_list):
            # get the add_tuple
            add_tuple = pq.peek_smallest()[0]
            # add vertex
            temporary_solution = add_vertex(
                OP_formulation, new_solution, add_tuple)
            # if route length is within constraint
            if temporary_solution['route_length'] < OP_formulation.graph['maximum_path_length']:
                # set the infeasible flag to false
                infeasible = False
                # overwrite the new solution with the temporary solution
                new_solution = temporary_solution
                # remove the priority queue
                del pq_list[pq_idx]
                # break
                break
        # if infeasible
        if infeasible == True:
            # return the new solution
            return new_solution
        # unpack the add_tuple
        prev_vertex, added_vertex, next_vertex = add_tuple
        # iterate through the list of priority queues
        for pq_idx, (_, curr_vertex, benefit) in enumerate(pq_list):
            # remove the affected tuples
            affected_tuple = (prev_vertex, curr_vertex, next_vertex)
            pq_list[pq_idx][0].remove_item(affected_tuple)
            # set the new tuples
            new_tuple = (prev_vertex, curr_vertex, added_vertex)
            # compute the cost
            cost = 0.
            cost += OP_formulation.edges[prev_vertex,
                                         curr_vertex]['distance']
            cost += OP_formulation.edges[curr_vertex,
                                         added_vertex]['distance']
            cost -= OP_formulation.edges[prev_vertex,
                                         added_vertex]['distance']
            # compute the negative bcr
            # NOTICE! It is negative because the pq is a min heap
            negative_bcr = -benefit/cost
            # add the add tuple to the heapdict
            pq_list[pq_idx][0].set_item(new_tuple, negative_bcr)
            # set another new tuples
            new_tuple = (added_vertex, curr_vertex, next_vertex)
            # compute the cost
            cost = 0.
            cost += OP_formulation.edges[added_vertex,
                                         curr_vertex]['distance']
            cost += OP_formulation.edges[curr_vertex,
                                         next_vertex]['distance']
            cost -= OP_formulation.edges[added_vertex,
                                         next_vertex]['distance']
            # compute the negative bcr
            # NOTICE! It is negative because the pq is a min heap
            negative_bcr = -benefit/cost
            # add the add tuple to the heapdict
            pq_list[pq_idx][0].set_item(new_tuple, negative_bcr)
    return new_solution


# ---- solution_from_visited_vertices(OP_formulation, visited_vertices)
def solution_from_visited_list(OP_formulation, visited_vertices):
    """
    Create a solution from a visited list. The route length and objectives is 
    computed from scratch.

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including vertices, scores, 
        costs.
    visited_vertices : list
        A list of visited vertices with order.

    Returns
    -------
    solution : dict
        A dict with keys permutation, objectives and route length.

    """
    # create the default permutation, route length and objectives
    permutation = np.arange(len(OP_formulation))
    route_length = 0
    objectives = np.zeros_like(OP_formulation.nodes[0]['objectives'])
    # iterate through the list of visited vertices except for the last one
    for idx in range(len(visited_vertices)-1):
        curr_vertex = visited_vertices[idx]
        next_vertex = visited_vertices[idx+1]
        # change the permutation
        permutation[curr_vertex] = next_vertex
        # increase the route length
        route_length += OP_formulation.edges[curr_vertex,
                                             next_vertex]['distance']
        # increase the objectives
        objectives += OP_formulation.nodes[curr_vertex]['objectives']
    # check if it is a route(start != end) or a tour(start=end)
    if visited_vertices[0] != visited_vertices[-1]:
        permutation[visited_vertices[-1]] = -1
        objectives += OP_formulation.nodes[visited_vertices[-1]]['objectives']
    # create the solution
    solution = {'permutation': permutation,
                'route_length': route_length, 'objectives': objectives}
    return solution


# ---- add_vertex(old_solution): new_solution
def add_vertex(OP_formulation, old_solution, add_tuple):
    """
    add vertex to the old solution
    update route length based on the change
    update objectives

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including vertices, scores, 
        costs.
    old_solution : ndarray
        input solution.
    add_tuple : tuple
        a tuple (prev_vertex, curr_vertex, next_vertex) indicating the vertex 
        and the position to be added.

    Returns
    -------
    new_solution : dict
        output solution.

    """
    # copy the solution
    new_solution = deepcopy(old_solution)
    # unpack the tuple
    prev_vertex, curr_vertex, next_vertex = add_tuple
    # set the permutation
    new_solution['permutation'][prev_vertex] = curr_vertex
    new_solution['permutation'][curr_vertex] = next_vertex
    # set the route length
    new_solution['route_length'] -= OP_formulation.edges[prev_vertex,
                                                         next_vertex]['distance']
    new_solution['route_length'] += OP_formulation.edges[prev_vertex,
                                                         curr_vertex]['distance']
    new_solution['route_length'] += OP_formulation.edges[curr_vertex,
                                                         next_vertex]['distance']
    # set the objectives
    new_solution['objectives'] += OP_formulation.nodes[curr_vertex]['objectives']
    return new_solution


# ---- drop_vertex(old_solution): new_solution
def drop_vertex(OP_formulation, old_solution, drop_tuple):
    """
    remove vertex front the visited list of a solution
    add it to the non-visited set

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including vertices, scores, 
        costs.
    old_solution : dict
        input solution.
    drop_tuple : tuple
        a tuple (prev_vertex, curr_vertex, next_vertex) indicating the vertex 
        and the position to be dropped.

    Returns
    -------
    new_solution : dict
        output solution.

    """
    # copy the solution
    new_solution = deepcopy(old_solution)
    # unpack the drop_tuple
    prev_vertex, curr_vertex, next_vertex = drop_tuple
    # set the permutation
    new_solution['permutation'][prev_vertex] = next_vertex
    new_solution['permutation'][curr_vertex] = curr_vertex
    # set the route_length
    new_solution['route_length'] += OP_formulation.edges[prev_vertex,
                                                         next_vertex]['distance']
    new_solution['route_length'] -= OP_formulation.edges[prev_vertex,
                                                         curr_vertex]['distance']
    new_solution['route_length'] -= OP_formulation.edges[curr_vertex,
                                                         next_vertex]['distance']
    # set the objectives
    new_solution['objectives'] -= OP_formulation.nodes[curr_vertex]['objectives']
    return new_solution


# ---- local_search(OP_formulation, weight): local_pareto_set_appr
def local_search(OP_formulation, solution, weight, maximum_trial_number, pareto_set_approximation, dropped_proportion):
    """
    Perform local search iteratively until trial_number hits maximum_trial_number

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including vertices, scores, 
        costs.
    solution : dict
        A solution where the local search start from.
    weight : ndarray
        A point on a standard (K-1)-simplex guiding the search.
    maximum_trial_number : int
        The number of trials before giving up searching.
    pareto_set_approximation : dict
        A dict of non-dominated solutions, with the key-value pair 
        of permutation-solution.

    Returns
    -------
    pareto_set_approximation : dict
        A dict of non-dominated solutions, with the key-value pair 
        of permutation-solution.

    """
    trial_num = 0
    while trial_num < maximum_trial_number:
        old_weighted_objectives = weight@solution['objectives']
        solution = shorten(OP_formulation, solution)
        solution = drop(OP_formulation, solution, dropped_proportion, weight)
        solution = add(OP_formulation, solution, weight)
        new_weighted_objectives = weight@solution['objectives']
        pareto_set_approximation, updated = pareto_update(
            pareto_set_approximation, solution)

        if updated == True:
            trial_num = 0
        elif new_weighted_objectives <= old_weighted_objectives:
            trial_num += 1

    return pareto_set_approximation


# ---- default_solution(OP_formulation)
def default_solution(OP_formulation, weight):
    """
    Generate a new solution by applying the add operator to a empty solution 
    which only contains start and end vertices.

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including vertices, scores, 
        costs.
    weight : ndarray
        A point on a standard (K-1)-simplex guiding the search.

    Returns
    -------
    solution : dict
        A solution generated.

    """
    # create a empty solution
    visited_vertices = [OP_formulation.graph['start'],
                        OP_formulation.graph['end']]
    solution = solution_from_visited_list(OP_formulation, visited_vertices)
    # apply the add operator to the empty solution
    solution = add(OP_formulation, solution, weight)
    return solution


# ---- IBEA4MOOP
def IBEA4MOOP(OP_formulation, maximum_iteration, maximum_trial_number, dropped_proportion):
    # Initialize the Pareto front approximation as a empty dict
    pareto_set_approximation = dict()

    # get basic info from the problem
    K = len(OP_formulation.nodes[0]['objectives'])

    # for each standard bases e_k
    for k in range(K):
        # local search with the weight e_k
        weight = np.zeros(K)
        weight[k] = 1
        # create child with add operator
        solution = default_solution(OP_formulation, weight)
        pareto_set_approximation = local_search(
            OP_formulation, solution, weight, maximum_trial_number, pareto_set_approximation, dropped_proportion)

    # while terminal condition not met
    for itr in range(maximum_iteration):
        # select
        parents = select(pareto_set_approximation)
        # crossover
        child, weight = crossover(OP_formulation, parents)
        # local search with the weight of the child solution
        pareto_set_approximation = local_search(
            OP_formulation, child, weight, maximum_trial_number, pareto_set_approximation, dropped_proportion)

    # return result
    pareto_front_approximation = []
    for solution in pareto_set_approximation.values():
        pareto_front_approximation.append(solution['objectives'])
    pareto_set_approximation = list(pareto_set_approximation)

    return pareto_front_approximation, pareto_set_approximation


# %% initialization
# global seed
rng = np.random.default_rng(0)

# ignore 0 division warning
np.seterr(divide='ignore')

# find the paths of all the txt files
txt_paths = list(
    Path("../../../dataset/moop/2 objectives/dmatrix").rglob("2_p97_t*.[tT][xX][tT]"))

# set params
maximum_trial_number = 10
maximum_iteration = 10
runs_num = 10
dropped_proportion = 0.2

# %% problem formulation
for txt_path in txt_paths:
    print(txt_path.stem)
    # init figure
    plt.figure()
    plt.xlabel('obj_1')
    plt.ylabel('obj_2')
    plt.title('Pareto front approximations')
    # clear file
    open('results/'+txt_path.stem+'_MOOP_front', 'w').close()
    open('results/'+txt_path.stem+'_MOOP_set', 'w').close()
    for run in range(runs_num):
        df = pd.read_csv(txt_path, comment='/',
                         names=list(range(2144)), on_bad_lines='skip', index_col=0)
        OP_formulation = formulate_OP(df)
        pareto_front_approximation, pareto_set_approximation = IBEA4MOOP(
            OP_formulation, maximum_iteration, maximum_trial_number, dropped_proportion)
        # write files
        pareto_front_approximation = pd.DataFrame(pareto_front_approximation)
        pareto_set_approximation = pd.DataFrame(pareto_set_approximation)
        pareto_front_approximation_csv = pareto_front_approximation.to_csv(
            index=False, header=False, sep=' ')
        pareto_set_approximation_csv = pareto_set_approximation.to_csv(
            index=False, header=False)
        with open('results/'+txt_path.stem+'_MOOP_front', 'a') as file:
            file.write(pareto_front_approximation_csv+'\n')
        with open('results/'+txt_path.stem+'_MOOP_set', 'a') as file:
            file.write(pareto_set_approximation_csv+'\n')
        # plot fronts
        if run < 5:
            pareto_front_approximation = pareto_front_approximation.sort_values(
                by=0)
            plt.step(
                pareto_front_approximation[0], pareto_front_approximation[1], 'o-')
    plt.savefig('results/'+txt_path.stem+'_MOOP_front.jpg')
    plt.close()

"""
comment:
    my new optimizer to the MOOP
"""
