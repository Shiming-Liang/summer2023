#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:10:12 2023

@author: shiming
"""

# %% imports
import rasterio
import matplotlib.pyplot as plt

# %% Opening a dataset in reading mode
dataset = rasterio.open('USGS_13_n38w119_20230308.tif')
print(f'name: {dataset.name}')
print(f'mode: {dataset.mode}')
print(f'mode: {dataset.closed}')

# %% Dataset attributes
print(f'count: {dataset.count}')
print(f'width: {dataset.width}')
print(f'height: {dataset.height}')
print('index: dtype:', {i: dtype for i,
      dtype in zip(dataset.indexes, dataset.dtypes)})

# %% Dataset georeferencing
print(f'bounds: {dataset.bounds}')
print(f'transform: {dataset.transform}')
print(f'(row, col)->(x, y): (0, 0)->{dataset.transform * (0, 0)}')
print(
    f'(row, col)->(x, y): (dataset.width, dataset.height)->{dataset.transform * (dataset.width, dataset.height)}')
print(f'coordinate reference system (CRS): {dataset.crs}')

# %% Reading raster data
print(f'indices: {dataset.indexes}')
band1 = dataset.read(1)
print(band1[dataset.height // 2, dataset.width // 2])

# %% Spatial indexing
x, y = (dataset.bounds.left + 0.1, dataset.bounds.top - 0.1)
row, col = dataset.index(x, y)
print(band1[row, col])
print(dataset.xy(dataset.height // 2, dataset.width // 2))

# %% plot heatmap
# Plotting the heatmap
extent = list(dataset.bounds)
extent[1], extent[2] = extent[2], extent[1]
plt.figure(figsize=(8, 8))  # Set the size of the figure (optional)
plt.imshow(band1, extent=extent)
plt.colorbar(label='Elevation/meter')
plt.title('The Whole Tile')
plt.xlabel('Longitude/degree')
plt.ylabel('Latitude/degree')
plt.savefig('whole_tile.jpg')

# Plotting the heatmap for the sage hen area
left, right, bottom, top = -118.19, -118.13, 37.46, 37.52
idx_bottom, idx_left = dataset.index(left, bottom)
idx_top, idx_right = dataset.index(right, top)
sagehen = band1[idx_top:idx_bottom, idx_left:idx_right]
plt.figure(figsize=(8, 8))  # Set the size of the figure (optional)
plt.imshow(sagehen, extent=(left, right, bottom, top))
plt.colorbar(label='Elevation/meter')
plt.title('Sage Hen')
plt.xlabel('Longitude/degree')
plt.ylabel('Latitude/degree')
plt.ticklabel_format(useOffset=False)
plt.savefig('sage_hen.jpg')
