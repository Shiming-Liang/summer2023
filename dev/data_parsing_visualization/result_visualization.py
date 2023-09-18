#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 20:36:02 2023

@author: shiming
"""

# imports
import glob
import os
import csv
import pandas as pd
from matplotlib import pyplot as plt

# container for all paths
paths = {'IBEA': {}, 'ACO': {}, 'VNS': {}}

# get all the file path by keyword: *_IBEA
directory = "./"  # Replace this with the directory you want to search in
suffix = "_IBEA"
paths_IBEA = glob.glob(f"{directory}/**/*{suffix}", recursive=True)

# get the problem name
problems = set()
for path_IBEA in paths_IBEA:
    filename = os.path.basename(path_IBEA)
    problem = filename[:-5]
    problems.add(problem)
    paths['IBEA'][problem] = path_IBEA
problems = list(problems)

# get the file path by keyword: problem name
for problem in problems:
    path_ACO = glob.glob(f"{directory}/**/{problem}_ACO", recursive=True)
    path_VNS = glob.glob(f"{directory}/**/{problem}_VNS", recursive=True)
    paths['ACO'][problem] = path_ACO[0]
    paths['VNS'][problem] = path_VNS[0]


# read and parse the files
def read_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=' ')

        current_section = []
        all_sections = []

        for row in reader:
            if row:  # Non-empty line (i.e., a data row)
                current_section.append(row)
            else:  # Empty line (i.e., a section separator)
                if current_section:  # Add the current section to all_sections if it's not empty
                    all_sections.append(current_section)
                current_section = []  # Reset the current section

        # Don't forget the last section if the file doesn't end with an empty line
        if current_section:
            all_sections.append(current_section)

    return all_sections


# iterate through problems
for problem in problems:
    # get the path
    path_ACO = paths['ACO'][problem]
    path_IBEA = paths['IBEA'][problem]
    path_VNS = paths['VNS'][problem]

    # parse csv files
    result_ACO = read_csv(path_ACO)
    result_IBEA = read_csv(path_IBEA)
    result_VNS = read_csv(path_VNS)

    # init figure
    plt.figure()
    plt.xlabel('obj_1')
    plt.ylabel('obj_2')
    plt.title('Pareto front approximations')

    # for each run
    for run in range(10):
        # get the result of the run
        run_result_ACO = pd.DataFrame(result_ACO[run]).astype(float)
        run_result_IBEA = pd.DataFrame(result_IBEA[run]).astype(float)
        run_result_VNS = pd.DataFrame(result_VNS[run]).astype(float)

        # sort the results by the first objective
        run_result_ACO = run_result_ACO.sort_values(by=0)
        run_result_IBEA = run_result_IBEA.sort_values(by=0)
        run_result_VNS = run_result_VNS.sort_values(by=0)

        # step plot with different colors indicating methods
        plt.step(run_result_ACO[0], run_result_ACO[1],
                 'ro-', alpha=0.2, label='P-ACO')
        plt.step(run_result_IBEA[0], run_result_IBEA[1],
                 'go-', alpha=0.2, label='IBEA')
        plt.step(run_result_VNS[0], run_result_VNS[1],
                 'bo-', alpha=0.2, label='P-VNS')
    # Get existing handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Remove duplicate labels and handles
    unique = [(h, l) for i, (h, l) in enumerate(
        zip(handles, labels)) if l not in labels[:i]]
    unique_handles, unique_labels = zip(*unique)

    # Create legend
    plt.legend(unique_handles, unique_labels)
    plt.savefig('comparison_plots/'+problem)
    plt.close()
