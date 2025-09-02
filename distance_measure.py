import numpy as np

"""
n = total data points, k = cluster count, d = dimesion count
Function: Calculates Euclidian distance between clusters and all data points
Parameters: Takes in 2 matrices: a = (n) x (d) and b = (k) x (d)
Return: The Euclidian distance matrix (n) x (k)
"""
def euclidean_distance(a, b):
    # Dimension expansion allows for numpy broadcasting to calcuate all at once rather than one by one
    expand_a = a[:, np.newaxis, :]
    expand_b = b[np.newaxis, :, :]

    expand_distance = (expand_a - expand_b) ** 2
    distance = np.sqrt(expand_distance.sum(axis = 2))
    return distance