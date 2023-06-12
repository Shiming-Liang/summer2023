#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:47:48 2023

@author: shiming
"""

# %% imports
import pandas as pd
import numpy as np
# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# %% constants
# rng = np.random.RandomState(0)
rng = np.random.RandomState()
regressor = RandomForestRegressor(random_state=0)

# %% load dataset
# load from xlsx
df = pd.read_excel('../../dataset/SHF_Uncertainty.xlsx')
# turn dataframe into array
raw = df.to_numpy()
# take the last 5 col: ['Attachedness', 'Lithology', 'Geometry', 'lat', 'long']
raw_digit = raw[:, -5:].astype(float)
# remove all nan rows
raw_digit_filtered = raw_digit[~np.isnan(raw_digit[:, :3]).all(axis=1)]


# %% imputations train-val
# get full rows
data_full = raw_digit_filtered[~np.isnan(
    raw_digit_filtered[:, :3]).any(axis=1)]


# add fake nan
def add_missing_values(data_full):
    n_samples, n_features = data_full[:, :3].shape

    # Add missing values in 75% of the lines
    missing_rate = 0.5
    n_missing_samples = int(n_samples * missing_rate)

    missing_samples = np.zeros(n_samples, dtype=bool)
    missing_samples[:n_missing_samples] = True

    rng.shuffle(missing_samples)
    missing_features = rng.choice(
        n_features, n_missing_samples, p=[0, 0.2, 0.8])
    data_miss = data_full.copy()
    data_miss[missing_samples, missing_features] = np.nan

    return data_miss


data_miss = add_missing_values(data_full)


# try the impuation methods and get scores
def get_scores_for_imputer(imputer, data_miss, data_full):
    data_impute = imputer.fit_transform(data_miss)
    data_impute[:, :3] = np.round(data_impute[:, :3])
    impute_scores = mean_squared_error(data_full, data_impute)
    return impute_scores


# initiate scores
scores = []
x_labels = []


# full data as baseline
def get_full_score(data_full):
    full_scores = mean_squared_error(data_full, data_full)
    return full_scores


scores.append(get_full_score(data_full))
x_labels.append('full')


# kNN-imputation
def get_impute_knn_score(data_miss, data_full):
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=5)
    data_miss_scale = data_miss.copy()
    data_miss_scale[:, -2:] *= 10000
    data_impute_scale = imputer.fit_transform(data_miss_scale)
    data_impute = data_impute_scale
    data_impute[:, -2:] /= 10000
    data_impute[:, :3] = np.round(data_impute[:, :3])
    knn_impute_scores = mean_squared_error(data_full, data_impute)
    # knn_impute_scores = get_scores_for_imputer(imputer, data_miss, data_full)
    return knn_impute_scores


scores.append(get_impute_knn_score(data_miss, data_full))
x_labels.append('knn')


# KNeighborsRegressor
def get_impute_itr_knn_score(data_miss, data_full):
    estimator = KNeighborsRegressor(n_neighbors=5)
    tol = 1e-2
    imputer = IterativeImputer(
        random_state=0, estimator=estimator, max_iter=100, tol=tol
    )
    data_miss_scale = data_miss.copy()
    data_miss_scale[:, -2:] *= 10000
    data_impute_scale = imputer.fit_transform(data_miss_scale)
    data_impute = data_impute_scale
    data_impute[:, -2:] /= 10000
    data_impute[:, :3] = np.round(data_impute[:, :3])
    knn_impute_scores = mean_squared_error(data_full, data_impute)
    # knn_impute_scores = get_scores_for_imputer(imputer, data_miss, data_full)
    return knn_impute_scores


scores.append(get_impute_itr_knn_score(data_miss, data_full))
x_labels.append('iterative knn')


# iterative imputation
def get_impute_mean(data_miss, data_full):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    iterative_impute_scores = get_scores_for_imputer(
        imputer, data_miss, data_full)
    return iterative_impute_scores


scores.append(get_impute_mean(data_miss, data_full))
x_labels.append('mean')


# plot results
plt.figure()
n_bars = len(scores)
xval = np.arange(n_bars)

# plot results
ax1 = plt.subplot()
for j in xval:
    ax1.barh(
        j,
        scores[j],
        alpha=0.6,
        align="center",
    )

ax1.set_title("Imputation Techniques with Sage Hen Data")
ax1.set_xlim(left=np.min(scores) * 0.9,
             right=np.max(scores) * 1.1)
ax1.set_yticks(xval)
ax1.set_xlabel("MSE")
ax1.invert_yaxis()
ax1.set_yticklabels(x_labels)

plt.show()


# %% repeated trials
# get full rows
data_full = raw_digit_filtered[~np.isnan(
    raw_digit_filtered[:, :3]).any(axis=1)]

num_trial = 1000
scores = np.zeros(4)
for trial in range(num_trial):
    data_miss = add_missing_values(data_full)
    scores[0] += get_full_score(data_full)/num_trial
    scores[1] += get_impute_knn_score(data_miss, data_full)/num_trial
    # scores[2] += get_impute_itr_forest_score(data_miss, data_full)/num_trial
    scores[2] += get_impute_itr_knn_score(data_miss, data_full)/num_trial
    scores[3] += get_impute_mean(data_miss, data_full)/num_trial


# plot results
plt.figure()
n_bars = len(scores)
xval = np.arange(n_bars)

# plot results
ax1 = plt.subplot()
for j in xval:
    ax1.barh(
        j,
        scores[j],
        alpha=0.6,
        align="center",
    )

ax1.set_title("Imputation Techniques with Sage Hen Data")
ax1.set_xlim(left=np.min(scores) * 0.9,
             right=np.max(scores) * 1.1)
ax1.set_yticks(xval)
ax1.set_xlabel("MSE")
ax1.invert_yaxis()
ax1.set_yticklabels(x_labels)

plt.show()
plt.savefig('knn_pos_only.jpg')
