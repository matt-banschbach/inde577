import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from em_el.utils import euclidean
import numpy as np


def euclidean(point, data):
    """
    Return euclidean distances between a point & a dataset
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))


class KMeans:
    """
    KMeans clustering algorithm implementation.

    Parameters:
    k (int): Number of clusters.
    max_iter (int): Maximum number of iterations for the algorithm to run.
    """

    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def initialize_centroids(self, X):
        """
        Initialize centroids using the k-means++ method.

        Parameters:
        X (np.ndarray): The dataset.

        Returns:
        np.ndarray: Initial centroids.
        """
        # Pick a random point from train data for the first centroid
        centroids = [X[np.random.choice(len(X))]]

        for _ in range(self.k - 1):
            # Calculate distances from points to the centroids
            dists = np.min(euclidean(X, centroids), axis=1)
            # Normalize the distances
            dists /= np.sum(dists)
            # Choose remaining points based on their distances
            new_centroid_idx = np.random.choice(range(len(X)), size=1, p=dists)[0]
            centroids.append(X[new_centroid_idx])

        return np.array(centroids)

    def fit(self, X):
        """
        Fit the KMeans algorithm to the dataset.

        Parameters:
        X (np.ndarray): The dataset.
        """
        self.centroids = self.initialize_centroids(X)

        for iter in range(self.max_iter):
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.k)]

            for x in X:
                dists = euclidean([x], self.centroids)[0]
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) if cluster else prev_centroids[i] for i, cluster in enumerate(sorted_points)]

            # Check for convergence
            if np.all(prev_centroids == self.centroids):
                break

        self.sorted_points = sorted_points

    def evaluate(self, X):
        """
        Evaluate the dataset by assigning each point to the nearest centroid.

        Parameters:
        X (np.ndarray): The dataset.

        Returns:
        tuple: Centroids and their corresponding indices for each point.
        """
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean([x], self.centroids)[0]
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)

        return centroids, centroid_idxs

    def inertia(self):
        """
        Calculate the Within-Cluster Sum of Squares (WCSS).

        Returns:
        float: WCSS value.
        """
        wcss = 0
        for i in range(self.k):
            cluster_points = np.array(self.sorted_points[i])
            centroid = self.centroids[i]
            wcss += np.sum((cluster_points - centroid) ** 2)
        return wcss

    def classification_error(self, X, y):
        """
        Calculate the classification error.

        Parameters:
        X (np.ndarray): The dataset.
        y (np.ndarray): True labels.

        Returns:
        float: Classification error rate.
        """
        _, centroid_idxs = self.evaluate(X)
        return np.sum(centroid_idxs != y) / len(y)



class DBSCAN:
    """
    Density-based
    """