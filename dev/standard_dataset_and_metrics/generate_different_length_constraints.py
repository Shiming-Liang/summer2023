#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:56:59 2023

@author: shiming
"""

# imports
from pathlib import Path

# get the paths of files in t1_t20
original_paths = list(
    Path("../../dataset/moop/2 objectives/dmatrix/t1_t20").rglob("*.[tT][xX][tT]"))
# loop through the paths
for original_path in original_paths:
    # read in a original file
    with open(original_path, 'r') as file:
        original_data = file.read()
    # for t in range(1, 21)
    for t in range(1, 21):
        # replace text
        new_data = original_data.replace(
            'U, -1, 1', f'U, -1, {t}')
        # write file
        new_filepath = original_path.with_name(
            original_path.stem+f'_t{t:03}.txt')
        with open(new_filepath, 'w') as file:
            file.write(new_data)

# get the paths of files in t10_t150
original_paths = list(
    Path("../../dataset/moop/2 objectives/dmatrix/t10_t150").rglob("*.[tT][xX][tT]"))
# loop through the paths
for original_path in original_paths:
    # read in a original file
    with open(original_path, 'r') as file:
        original_data = file.read()
    # for t in range(1, 21)
    for t in range(10, 155, 5):
        # replace text
        new_data = original_data.replace('U, -1, 10', f'U, -1, {t}')
        # write file
        new_filepath = original_path.with_name(
            original_path.stem+f'_t{t:03}.txt')
        with open(new_filepath, 'w') as file:
            file.write(new_data)
