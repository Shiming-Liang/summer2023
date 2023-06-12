#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:47:48 2023

@author: shiming
"""

# %% imports
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# %% constants
# rng = np.random.RandomState(0)
rng = np.random.RandomState()

# %% load dataset
# load from xlsx
df = pd.read_excel('../../dataset/SHF_Uncertainty.xlsx')
# turn dataframe into array
raw = df.to_numpy()
# take the last 5 col: ['Attachedness', 'Lithology', 'Geometry', 'lat', 'long']
raw_digit = raw[:, -5:].astype(float)
# remove all nan rows
data_miss = raw_digit[~np.isnan(raw_digit[:, :3]).all(axis=1)]

# %% impute nan with knn imputer
# scaler = MinMaxScaler()
# imputer = KNNImputer()

# data_miss_scale = scaler.fit_transform(data_miss)
# data_impute_scale = imputer.fit_transform(data_miss_scale)
# data_impute = scaler.inverse_transform(data_impute_scale)
# data_impute[:, :3] = np.round(data_impute[:, :3])

imputer = KNNImputer()

data_miss_scale = data_miss.copy()
data_miss_scale[:, :3] /= 5
data_miss_scale[:, -2:] *= 20
data_impute_scale = imputer.fit_transform(data_miss_scale)
data_impute = data_impute_scale.copy()
data_impute[:, :3] *= 5
data_impute[:, -2:] /= 20
data_impute[:, :3] = np.round(data_impute[:, :3])

# %% plot
# Attachedness
# remove nan to avoid legend error
data_atta = data_miss[~np.isnan(data_miss[:, 0])]

fig = plt.figure()
fig.suptitle('Attachedness')
ax1 = plt.subplot(121)
scatter = ax1.scatter(
    data_impute[:, 4], data_impute[:, 3], c=data_impute[:, 0])
ax1.set_xlabel('long')
ax1.set_ylabel('lat')
ax1.axis('equal')
ax1.legend(*scatter.legend_elements(), title="Uncertainty")
ax2 = plt.subplot(122)
scatter = ax2.scatter(data_atta[:, 4], data_atta[:, 3], c=data_atta[:, 0])
ax2.set_xlabel('long')
ax2.set_ylabel('lat')
ax2.axis('equal')
ax2.legend(*scatter.legend_elements(), title="Uncertainty")
plt.savefig('Attachedness.jpg')

# Lithology
# remove nan to avoid legend error
data_lith = data_miss[~np.isnan(data_miss[:, 1])]

fig = plt.figure()
fig.suptitle('Lithology')
ax1 = plt.subplot(121)
scatter = ax1.scatter(
    data_impute[:, 4], data_impute[:, 3], c=data_impute[:, 1])
ax1.set_xlabel('long')
ax1.set_ylabel('lat')
ax1.axis('equal')
ax1.legend(*scatter.legend_elements(), title="Uncertainty")
ax2 = plt.subplot(122)
scatter = ax2.scatter(data_lith[:, 4], data_lith[:, 3], c=data_lith[:, 1])
ax2.set_xlabel('long')
ax2.set_ylabel('lat')
ax2.axis('equal')
ax2.legend(*scatter.legend_elements(), title="Uncertainty")
plt.savefig('Lithology.jpg')

# Geometry
# remove nan to avoid legend error
data_geom = data_miss[~np.isnan(data_miss[:, 2])]

fig = plt.figure()
fig.suptitle('Geometry')
ax1 = plt.subplot(121)
scatter = ax1.scatter(
    data_impute[:, 4], data_impute[:, 3], c=data_impute[:, 2])
ax1.set_xlabel('long')
ax1.set_ylabel('lat')
ax1.axis('equal')
ax1.legend(*scatter.legend_elements(), title="Uncertainty")
ax2 = plt.subplot(122)
scatter = ax2.scatter(data_geom[:, 4], data_geom[:, 3], c=data_geom[:, 2])
ax2.set_xlabel('long')
ax2.set_ylabel('lat')
ax2.axis('equal')
ax2.legend(*scatter.legend_elements(), title="Uncertainty")
plt.savefig('Geometry.jpg')
