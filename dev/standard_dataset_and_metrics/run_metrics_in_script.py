#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:00:33 2023

@author: shiming
"""

# import
import pandas as pd
import subprocess

# set the input file
data = [[1, 0], [0.5, 0.5], [0, 1], [1.5, 0.75]]
df = pd.DataFrame(data)
df.to_csv('data_file', index=False, sep=' ')

# run the metric
args = ("metrics/hyp_ind", "hyp_ind_param.txt",
        "data_file", "reference_set", "output_file")
output = subprocess.check_call(args)

# get the result
file_in = open('output_file', 'r')
result = float(file_in.readline())
