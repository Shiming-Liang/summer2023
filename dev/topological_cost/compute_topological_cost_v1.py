#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 19:57:48 2023

@author: shiming
"""

# %% imports
import rasterio
from astar import AStar
from math import cos, hypot, pi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.spatial import distance_matrix
import pickle


# %% global constants
DIST_CONSTANT_Y = cos(37.49*pi/180)  # lat ~= 37.49
DIST_CONSTANT_R = 6378137*pi/180  # radius of earth ~= 6378137m

# %% load and clip topological dataset
# load tiff
topo = rasterio.open('USGS_13_n38w119_20230308.tif')
topo_np = topo.read(1)
topo.close()


# %% define the a star class
def latlon_distance(n1, n2):
    # compute the planar distance
    dx = n1[0]-n2[0]
    dy = (n1[1]-n2[1])*DIST_CONSTANT_Y
    d_planar = DIST_CONSTANT_R*hypot(dx, dy)

    # include the elevation
    dz = n1[2]-n2[2]

    return hypot(d_planar, dz)


class MyAStar(AStar):
    # node has the form (long, lat, elevation)

    def __init__(self, topo, topo_np):
        # cache topology dataset for index / xy switching
        self.topo = topo
        self.topo_np = topo_np
        self.elevation_resolution = 100
        self.latlon_resolution = 10  # step size to find the neighbor

    def neighbors(self, n):
        # find the index
        row_c, col_c = self.topo.index(n[1], n[0])

        # init the list of neighbors
        neighbors_list = []

        # for row
        for row in range(row_c-self.latlon_resolution, row_c+self.latlon_resolution+1, self.latlon_resolution):
            # for col
            for col in range(col_c-self.latlon_resolution, col_c+self.latlon_resolution+1, self.latlon_resolution):
                # for elevation
                for elevation in [n[2]-self.elevation_resolution, n[2], n[2]+self.elevation_resolution]:
                    # skip identity
                    if row == row_c and col == col_c and elevation == n[2]:
                        continue
                    # check occupancy
                    actual_elevation = self.topo_np[row, col]
                    if elevation > actual_elevation:
                        # get lat long
                        long, lat = self.topo.xy(row, col)
                        # append the list of neighbors
                        neighbors_list.append((lat, long, elevation))

        return neighbors_list

    def distance_between(self, n1, n2):
        return latlon_distance(n1, n2)

    def heuristic_cost_estimate(self, current, goal):
        return latlon_distance(current, goal)

    def is_goal_reached(self, current, goal):
        # get index
        row_c, col_c = self.topo.index(current[1], current[0])
        row_g, col_g = self.topo.index(goal[1], goal[0])

        # check if equal
        if abs(row_c-row_g) < self.latlon_resolution:
            if abs(col_c-col_g) < self.latlon_resolution:
                if abs(current[2]-goal[2]) < 2*self.elevation_resolution:
                    return True


# %% parse sage hen dataset
def get_latlon():
    # load from xlsx
    df = pd.read_excel('../../dataset/SHF_Uncertainty.xlsx')
    # drop duplicate points by name
    df = df.drop_duplicates(subset=['StraboSpot Dataset Download: '])
    # turn dataframe into array
    raw = df.to_numpy()
    # take the last 5 col: ['Attachedness', 'Lithology', 'Geometry', 'lat', 'long']
    raw_digit = raw[:, -5:].astype(float)
    # remove all nan rows
    latlon = raw_digit[~np.isnan(raw_digit[:, :3]).all(axis=1)][:, -2:]

    return latlon


# %% build adjacency matrix
def get_path(start, goal):
    start_lat, start_long = start
    goal_lat, goal_long = goal

    start_row, start_col = topo.index(start_long, start_lat)
    goal_row, goal_col = topo.index(goal_long, goal_lat)
    start_elevation = topo_np[start_row, start_col]
    goal_elevation = topo_np[goal_row, goal_col]

    start = (start_lat, start_long, start_elevation)
    goal = (goal_lat, goal_long, goal_elevation)

    my_astar = MyAStar(topo, topo_np)
    path = list(my_astar.astar(start, goal))
    return path


def get_cost(path):
    cost = 0.
    for i in range(len(path)-1):
        cost += latlon_distance(path[i], path[i+1])
    return cost


def get_costs_starting_from(start_goals_tuple):
    start, goals = start_goals_tuple
    costs = []
    paths = []
    for goal in goals:
        path = get_path(start, goal)
        cost = get_cost(path)
        costs.append(cost)
        paths.append(path)
        print(f'start: {start}, goal: {goal}: , cost: {cost}')

    return costs, paths


def build_adjacency_matrix(latlon, reload_old_result):
    if reload_old_result:
        with open('tri_entries.pickle', 'rb') as f:
            tri_entries = pickle.load(f)
    else:
        sum_distance = np.sum(distance_matrix(latlon, latlon), 1)
        latlon = latlon[np.argsort(sum_distance)]

        start_goals_tuples = []
        for i, start in enumerate(latlon[:-1]):
            goals = latlon[i+1:]
            start_goals_tuples.append((start, goals))
        with Pool(15) as p:
            tri_entries = p.map(get_costs_starting_from, start_goals_tuples)
        # tri_entries = []
        # for start_goals_tuple in start_goals_tuples:
        #     tri_entries.append(get_costs_starting_from(start_goals_tuple))
        with open('data.pickle', 'wb') as f:
            pickle.dump(tri_entries, f, pickle.HIGHEST_PROTOCOL)

    adjacency_matrix = np.zeros((len(latlon), len(latlon)))
    for i, (costs, _) in enumerate(tri_entries):
        adjacency_matrix[i, i+1:] = costs
        adjacency_matrix[i+1:, i] = costs

    # Plotting the heatmap for the sage hen area
    left, right, bottom, top = -118.19, -118.13, 37.46, 37.52
    idx_bottom, idx_left = topo.index(left, bottom)
    idx_top, idx_right = topo.index(right, top)
    sagehen = topo_np[idx_top:idx_bottom, idx_left:idx_right]
    plt.figure(figsize=(8, 8))  # Set the size of the figure (optional)
    plt.imshow(sagehen, extent=(left, right, bottom, top))
    plt.colorbar(label='Elevation/meter')
    plt.title('Sage Hen')
    plt.xlabel('Longitude/degree')
    plt.ylabel('Latitude/degree')
    plt.ticklabel_format(useOffset=False)
    for _, paths in tri_entries:
        for path in paths:
            # plot route
            path = np.array(path)[:, :2]
            plt.plot(path[:, 1], path[:, 0])
    plt.savefig('sage_hen_astar.jpg')

    return adjacency_matrix


if __name__ == '__main__':
    latlon = get_latlon()
    adjacency_matrix = build_adjacency_matrix(latlon, False)
    with open('result.pickle', 'wb') as f:
        pickle.dump(adjacency_matrix, f, pickle.HIGHEST_PROTOCOL)
