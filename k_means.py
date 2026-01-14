import numpy as np
from distance_measure import euclidean_distance
    

class KMeans:
    """
    Parameters: k for cluster total, max_iters for iteration count, 
    tolerance for smallest distance a centroid can move, 
    and a function to calculate distance from the centroids
    """
    def __init__(self, k = 3, max_iters = 300, tolerance = 0.0001, distance_function = euclidean_distance):
        # Set random seed for reproducability
        self.seed = np.random.default_rng(seed=42)
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.distance_function = distance_function

        # No idea about dimensionality of data, so centroids and labels are set to None for now
        self.centroids = None
        self.labels = None

        # Iterations have not happened yet
        self.iterations = 0

    """
    Function: Clusters the data using the helper functions below
    Parameters: Takes in the data for the clustering to be ran on
    Return: The labels for each data point and the centroids, so the data can be graphed
    """
    def fit(self, X):
        min_bounds = np.min(X, axis = 0)
        max_bounds = np.max(X, axis = 0)
        self._initialize_centroids(min_bounds, max_bounds)
        self.iterations = 0

        while (self.iterations < self.max_iters):
            old_centroids = np.copy(self.centroids)
            self._assign_clusters(X)
            self._update_centroids(X, min_bounds, max_bounds)

            if not self._centroid_distance_calculation(old_centroids, self.centroids):
                return self.labels, self.centroids
        
        return self.labels, self.centroids


    """
    Function: Places the centroids in space
    Parameters: take in the minimum and maximum bounds of each dimension (will be a 1d matrix) 
    """
    def _initialize_centroids(self, min_bounds, max_bounds):
        # Generates a random number between the low and high bounds of each dimension
        # shape returns a tuple, so doing [0] will get the portion I want (since it will be of size dimensions x 1)
        self.centroids = self.seed.uniform(low = min_bounds, high = max_bounds, size = (self.k, min_bounds.shape[0]))

    """
    Function: Assign each piece of data to a centroid
    Parameters: The dataset we are working with
    """
    def _assign_clusters(self, X):
        distance = self.distance_function(X, self.centroids)

        # Gives the index smallest value in each row (column = 0, row = 1)
        # That value will become the centroid for each data point
        self.labels = np.argmin(distance, axis = 1)
        
    """
    Function: Relocate the centroids based on the mean of the data, or random if no data assigned
    Parameters: Takes in the dataset matrix
    """
    def _update_centroids(self, X, min_bounds, max_bounds):
        for cluster_id in range(self.k):
            new_location = X[self.labels == cluster_id]

            if new_location.size == 0: 
                self.centroids[cluster_id] = self.seed.uniform(low = min_bounds, high = max_bounds)
            else: 
                self.centroids[cluster_id] = np.mean(new_location, axis = 0)

    """
    Function: Calculate the distance between centroids to determine if more runs are necessary
    Parameters: Takes in the locations of new and old centroids in np arrays
    Return: A boolean indicating if the function needs to continue or not
    """
    def _centroid_distance_calculation(self, old_centroids, new_centroids):
        return np.sqrt(np.sum((new_centroids - old_centroids) ** 2)) > self.tolerance