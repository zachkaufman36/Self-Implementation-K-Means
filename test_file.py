import k_means
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

""" Generates sample data and runs the k-means algorithm on it, then graphs the results """
# Generate sample data
k = 3
X, y = make_blobs(centers=k, n_samples=500, n_features=2, shuffle=True, random_state=42)

# Run the k-means algorithm
kmeans = k_means.KMeans(k=k)
labels, centroids = kmeans.fit(X)

# Graph the results
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
