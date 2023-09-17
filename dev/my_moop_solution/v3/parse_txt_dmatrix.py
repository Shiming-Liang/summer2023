#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:25:37 2023

@author: shiming
"""

# %% import
from heapq import heappush, heappop
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
import elkai
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from tqdm import tqdm
# from line_profiler import profile
from py2opt.routefinder import RouteFinder


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
    OP_formulation = nx.DiGraph()
    OP_formulation.add_nodes_from(range(n))
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
            OP_formulation.add_edge(i, j, distance=distance_matrix[i, j])

    return OP_formulation


# ---- compute_fitness(objectives_of_all_solutions):
def compute_fitness(objectives_of_all_solutions):
    """
    Compute the fitness values of each solution according to IBEA.
    The method here is a bit different fron IBEA. Here we simply sum up the
    indicator values without taking the exponential.

    Parameters
    ----------
    objectives_of_all_solutions : nd array
        n*k array containing the objectives of n solutions.

    Returns
    -------
    fitness : nd array
        n array containing the fitness of n solutions.

    """
    # construct a empty array for fitness
    fitness = np.empty(len(objectives_of_all_solutions))
    # iterate through the objectives_of_all_solutions
    for solution_idx, objectives in enumerate(objectives_of_all_solutions):
        # compute sum(max(x2-x1)) and plug it in the fitness
        fitness[solution_idx] = (
            objectives-objectives_of_all_solutions).max(axis=1).sum()
    return fitness


# ---- select(pareto_set_approximation)
def select(population, maximum_population_size, tournament_size):
    """
    Select 2 parents using roulette wheel.
    The probability for each candidate is proportional to the exponential of 
    its hypervolume contribution.

    Parameters
    ----------
    population : dict
        A dict of solutions, with the key-value pair of permutation-solution.

    Returns
    -------
    parents : list
        maximum_population_size parents chosen by binary tournament.
    population : dict
        A dict of solutions, with the key-value pair of permutation-solution.

    """
    # get the objectives
    objectives_of_all_solutions = []
    for solution in population.values():
        objectives_of_all_solutions.append(solution['objectives'])
    objectives_of_all_solutions = np.array(objectives_of_all_solutions)
    # normalize the objectives
    scaler = MinMaxScaler()
    objectives_of_all_solutions = scaler.fit_transform(
        objectives_of_all_solutions)
    # compute the sum of binary epsilon indicators
    fitness = compute_fitness(objectives_of_all_solutions)
    # mating selection: find the parents by tournament selection
    population_key_list = list(population)
    parents = []
    for _ in range(maximum_population_size):
        tournament_candidate_indices = rng.choice(
            len(fitness), tournament_size)
        tournament_candidates_fitness = fitness[tournament_candidate_indices]
        tournament_winner_index = tournament_candidate_indices[tournament_candidates_fitness.argmax(
        )]
        parents.append(
            population[population_key_list[tournament_winner_index]])
    # environmental selection: drop the worst solutions
    if len(population) > maximum_population_size:
        keep_indices = np.argpartition(
            fitness, -maximum_population_size)[-maximum_population_size:]
        population = {
            population_key_list[keep_idx]: population[population_key_list[keep_idx]] for keep_idx in keep_indices}

    return parents, population


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


# ---- pareto_update(pareto_set_appr_0, pareto_set_appr_1): pareto_set_appr
def pareto_update(old_pareto_set_approximation, solution):
    """
    add the solution to the pareto_set_appr if it is Pareto efficient
    remove any solution dominated by the solution
    indicate Pareto efficiency
    add the dominated solution(s) to the backup pool

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
    key = tuple(solution['permutation'])
    if key in old_pareto_set_approximation:
        # if another_solution dominate solution, set False updated
        updated = False
        # break
        return old_pareto_set_approximation, updated

    # prepare for dominance comparision
    updated = True
    new_pareto_set_approximation = deepcopy(old_pareto_set_approximation)
    objectives = solution['objectives']

    # iterate through the pareto_set_appr
    for another_key, another_solution in old_pareto_set_approximation.items():
        another_objectives = another_solution['objectives']

        if (objectives >= another_objectives).all() and (objectives > another_objectives).any():
            # if solution dominate another_solution
            # del another_solution from the Pareto set approximation
            del new_pareto_set_approximation[another_key]

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

    # # throw the distance matrix into 2-opt
    # route_finder = RouteFinder(distance_matrix, range(len(node_list)),
    #                            iterations=1, return_to_begin=True, verbose=False)
    # _, tsp_solution = route_finder.solve()

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


# ---- drop(OP_formulation, old_solution): new_solution
def drop(OP_formulation, old_solution):
    """
    Perform the drop operator on the old solution.
    It drop the vertex most costy to be included in the solution until the 
    route length constraint is satisfied.

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including vertices, scores, 
    old_solution : dict
        The solution to be modified.

    Returns
    -------
    new_solution : dict
        The solution after modification.

    """
    # get the visited vertices
    visited_vertices = visited_vertices_from_solution(
        OP_formulation, old_solution)
    # if the number of vertices to be dropped is 0, return the old solution
    if len(visited_vertices) == 2:
        return old_solution
    # create a new solution
    new_solution = deepcopy(old_solution)
    # unpack the solution
    permutation = new_solution['permutation']
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
        # compute the negative_cost
        negative_cost = 0.
        negative_cost -= OP_formulation.edges[prev_vertex,
                                              curr_vertex]['distance']
        negative_cost -= OP_formulation.edges[curr_vertex,
                                              next_vertex]['distance']
        negative_cost += OP_formulation.edges[prev_vertex,
                                              next_vertex]['distance']
        # add the drop tuple to the heapdict
        # negative_cost because pq is a min heap
        # we want to drop the vertex with the largest cost
        pq.set_item(drop_tuple, negative_cost)
        # move to the next vertex
        prev_vertex = curr_vertex
        curr_vertex = next_vertex
        next_vertex = permutation[curr_vertex]
    # for i in range(number of vertices to be dropped)
    while pq.d:
        # pop the smallest one
        drop_tuple = pq.get_item()
        # drop_vertex
        new_solution = drop_vertex(OP_formulation, new_solution, drop_tuple)
        if new_solution['route_length'] < OP_formulation.graph['maximum_path_length']:
            return new_solution
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
            negative_cost = 0.
            negative_cost -= OP_formulation.edges[prev_prev_vertex,
                                                  prev_vertex]['distance']
            negative_cost -= OP_formulation.edges[prev_vertex,
                                                  next_vertex]['distance']
            negative_cost += OP_formulation.edges[prev_prev_vertex,
                                                  next_vertex]['distance']
            pq.set_item(new_tuple, negative_cost)
            new_tuple = (prev_vertex, next_vertex, next_next_vertex)
            negative_cost = 0.
            negative_cost -= OP_formulation.edges[prev_vertex,
                                                  next_vertex]['distance']
            negative_cost -= OP_formulation.edges[next_vertex,
                                                  next_next_vertex]['distance']
            negative_cost += OP_formulation.edges[prev_vertex,
                                                  next_next_vertex]['distance']
            pq.set_item(new_tuple, negative_cost)
        elif prev_vertex == OP_formulation.graph['start'] and next_vertex == OP_formulation.graph['end']:
            pass
        elif prev_vertex == OP_formulation.graph['start']:
            # remove the affected tuples
            next_next_vertex = permutation[next_vertex]
            affected_tuple = (curr_vertex, next_vertex, next_next_vertex)
            pq.remove_item(affected_tuple)
            # push in new tuple
            new_tuple = (prev_vertex, next_vertex, next_next_vertex)
            negative_cost = 0.
            negative_cost -= OP_formulation.edges[prev_vertex,
                                                  next_vertex]['distance']
            negative_cost -= OP_formulation.edges[next_vertex,
                                                  next_next_vertex]['distance']
            negative_cost += OP_formulation.edges[prev_vertex,
                                                  next_next_vertex]['distance']
            pq.set_item(new_tuple, negative_cost)
        else:
            # remove the affected tuples
            prev_prev_vertex = np.where(permutation == prev_vertex)[0][0]
            affected_tuple = (prev_prev_vertex, prev_vertex, curr_vertex)
            pq.remove_item(affected_tuple)
            # push in new tuple
            new_tuple = (prev_prev_vertex, prev_vertex, next_vertex)
            negative_cost = 0.
            negative_cost -= OP_formulation.edges[prev_prev_vertex,
                                                  prev_vertex]['distance']
            negative_cost -= OP_formulation.edges[prev_vertex,
                                                  next_vertex]['distance']
            negative_cost += OP_formulation.edges[prev_prev_vertex,
                                                  next_vertex]['distance']
            pq.set_item(new_tuple, negative_cost)
    return new_solution


# ---- non_visited_vertices_from_solution(OP_formulation, solution)
def non_visited_vertices_from_solution(OP_formulation, solution):
    """
    Compute the non-visited vertices from a solution.

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including vertices, scores, 
        costs.
    solution : dict
        The solution from which to get the visited vertices.

    Returns
    -------
    non_visited_vertices : list
        A list of non-visited vertices.

    """
    # take out the permutation for prettier look
    permutation = solution['permutation']
    # initialize the non-visited vertices
    non_visited_vertices = []
    # iterate through the permutation with enumerate
    for curr_vertex, next_vertex in enumerate(permutation):
        # if the vertex is unvisited
        if curr_vertex == next_vertex and curr_vertex != OP_formulation.graph['start']:
            non_visited_vertices.append(curr_vertex)
    return non_visited_vertices


# ---- add
def add(OP_formulation, old_solution, num_to_add):
    """
    Randomly add num_to_add bertices to the solution.

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including vertices, scores, 
    old_solution : dict
        The solution to be modified.
    num_to_add : int
        The number of vertices to be added at each mutation.

    Returns
    -------
    new_solution : dict
        The solution after modification.

    """
    # compute the non_visited_vertices
    non_visited_vertices = non_visited_vertices_from_solution(
        OP_formulation, old_solution)
    # compute the number of vertices to add
    actual_num_to_add = min(len(non_visited_vertices), num_to_add)
    # copy the old solution into a new one
    new_solution = deepcopy(old_solution)
    if actual_num_to_add > 0:
        # randomly pick a vertex
        curr_vertices = rng.choice(
            non_visited_vertices, actual_num_to_add, replace=False)
        # add the vertex after the start vertex
        prev_vertex = OP_formulation.graph['start']
        for curr_vertex in curr_vertices:
            next_vertex = new_solution['permutation'][prev_vertex]
            add_tuple = (prev_vertex, curr_vertex, next_vertex)
            new_solution = add_vertex(OP_formulation, new_solution, add_tuple)
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


# ---- default_solution(OP_formulation)
def default_solution(OP_formulation):
    """
    Generate a new solution by applying the add operator to a empty solution 
    which only contains start and end vertices.

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including vertices, scores, 
        costs.

    Returns
    -------
    solution : dict
        A solution generated.

    """
    # get the vertices except for start and end
    midway_vertices = list(OP_formulation.nodes())
    if OP_formulation.graph['start'] == OP_formulation.graph['end']:
        midway_vertices.remove(OP_formulation.graph['start'])
    else:
        midway_vertices.remove(OP_formulation.graph['start'])
        midway_vertices.remove(OP_formulation.graph['end'])
    # get a random half sample of the vertices
    half_midway_vertices = rng.choice(
        midway_vertices, int(len(midway_vertices)/2), replace=False)
    # create the solution
    visited_vertices = [OP_formulation.graph['start'], *
                        half_midway_vertices, OP_formulation.graph['end']]
    solution = solution_from_visited_list(OP_formulation, visited_vertices)
    return solution


# ---- initialize(pareto_set_approximation, population):
def initialize(OP_formulation, population, maximum_population_size):
    """
    Create a initial population by adding in trivial solutions.

    Parameters
    ----------
    population : dict
        A dict of solutions, with the key-value pair of permutation-solution.

    Returns
    -------
    population : dict
        A dict of solutions, with the key-value pair of permutation-solution.

    """
    for _ in range(maximum_population_size):
        solution = default_solution(OP_formulation)
        population[tuple(solution['permutation'])] = solution
    return population


# ---- mutate(OP_formulation, parents, population, pareto_set_approximation)
def mutate(OP_formulation, parents, population, pareto_set_approximation, num_to_add):
    """
    Evolve the parents, add the child to the population, update the Pareto 
    front approximation.

    Parameters
    ----------
    OP_formulation : Graph
        A networkx graph describing the OP problem, including vertices, scores, 
        costs.
    parents : list
        maximum_population_size parents chosen by binary tournament.
    population : dict
        A dict of solutions, with the key-value pair of permutation-solution.
    pareto_set_approximation : dict
        A dict of non-dominated solutions, with the key-value pair 
        of permutation-solution.
    num_to_add : int
        The number of vertices to be added at each mutation.

    Returns
    -------
    population : dict
        A dict of solutions, with the key-value pair of permutation-solution.
    pareto_set_approximation : dict
        A dict of non-dominated solutions, with the key-value pair 
        of permutation-solution.
    """
    # iterate through the parents
    for parent in parents:
        # apply the add operator
        child = add(OP_formulation, parent, num_to_add)
        # if route length constraint broken
        if child['route_length'] > OP_formulation.graph['maximum_path_length']:
            # apply the shorten operator
            child = shorten(OP_formulation, child)
        # if route length constraint still broken
        if child['route_length'] > OP_formulation.graph['maximum_path_length']:
            # apply the drop operator
            child = drop(OP_formulation, child)
        # add the child to the population
        key = tuple(child['permutation'])
        if key not in population:
            population[key] = child
        # perform pareto update
        pareto_set_approximation, _ = pareto_update(
            pareto_set_approximation, child)
    return population, pareto_set_approximation


# ---- IBEA4MOOP
def IBEA4MOOP(OP_formulation, maximum_iteration, maximum_population_size, tournament_size, add_proportion):
    # Initialize variables
    pareto_set_approximation = dict()
    population = dict()
    num_to_add = int(OP_formulation.graph['n']*add_proportion)

    # initialize the population
    population = initialize(OP_formulation, population,
                            maximum_population_size)

    # iteratively evolve the population
    for _ in range(maximum_iteration):
        # select
        parents, population = select(
            population, maximum_population_size, tournament_size)
        # mutate
        population, pareto_set_approximation = mutate(
            OP_formulation, parents, population, pareto_set_approximation, num_to_add)

    # return result
    pareto_front_approximation = []
    for solution in pareto_set_approximation.values():
        pareto_front_approximation.append(solution['objectives'])
    pareto_set_approximation = [(visited_vertices_from_solution(
        OP_formulation, solution), solution['route_length']) for solution in pareto_set_approximation.values()]

    return pareto_front_approximation, pareto_set_approximation


def main():
    # %% initialization
    # ignore 0 division warning
    np.seterr(divide='ignore')

    # find the paths of all the txt files
    txt_paths = list(
        Path("../../../dataset/moop/2 objectives/dmatrix").rglob("2_p273_*.[tT][xX][tT]"))

    # set params
    maximum_iteration = 20
    runs_num = 10
    maximum_population_size = 20
    tournament_size = 2
    add_proportion = 0.1

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

        # read and parse problem
        df = pd.read_csv(txt_path, comment='/',
                         names=list(range(2144)), on_bad_lines='skip', index_col=0)
        OP_formulation = formulate_OP(df)

        for run in tqdm(range(runs_num)):
            # run the algorithm
            pareto_front_approximation, pareto_set_approximation = IBEA4MOOP(
                OP_formulation, maximum_iteration, maximum_population_size, tournament_size, add_proportion)
            # write files
            pareto_front_approximation = pd.DataFrame(
                pareto_front_approximation)
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


# global seed
rng = np.random.default_rng(0)

if __name__ == "__main__":
    main()

"""
comment:
    my new optimizer to the MOOP
"""
