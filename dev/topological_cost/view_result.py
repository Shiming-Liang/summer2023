import numpy as np
import pickle

# FILEPATH: /home/shiming/Documents/summer2023/dev/topological_cost/view_result.py

# Load the pickle file
with open('result.pickle', 'rb') as file:
    result = pickle.load(file)

# Convert the loaded data to a NumPy array
adjacency_matrix = np.array(result)

# Save the NumPy array to a csv file
np.savetxt('adjacency_matrix.csv', adjacency_matrix, delimiter=',')