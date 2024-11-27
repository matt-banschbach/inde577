import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.eager.executor import Executor

from em_el.utils import euclidean

class KMeans:
    """
    K Means clustering
    """

    def __init__(self, k, max_iter=100):
        """
        Initializes a KMeans clustering object
        :param k: Number of clusters
        :param max_iter: Maximum number of algorithmic iterations before clustering algorithm terminates
        """

        self.classification = None
        self.sorted_points = None
        self.centroids = None
        self.k = k
        self.max_iter = max_iter
    
    @staticmethod
    def _initialize_centroids(k, X):
        """
        Initializes Centroids using the KMeans++ method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        :param k: Number of clusters
        :param X: (array-like) Feature Data
        :return: (list) the initial centroids
        """
        try:
            # Pick a random point from train data for first centroid
            centroids = [X[np.random.choice(len(X))]]  # TODO: More efficient way to do this?

            for _ in range(k - 1):  # We only need to set k-1 other centroids randomly
                # Calculate distances from points to the centroids
                dists = np.sum([euclidean(centroid, X) for centroid in centroids], axis=0)  # TODO: Why is this np.sum here?

                # Normalize the distances
                dists /= np.sum(dists)

                # Choose remaining points based on their distances; points further away have higher likelihood
                new_centroid_idx = np.random.choice(range(len(X)), size=1, p=dists)[0]  # Indexed @ zero to get val, not array of val
                centroids.append(X[new_centroid_idx])

            return centroids
        except Exception as e:
            print(f"Error during centroid initialization.")
            print(f"Exception: {e}")
            raise e


    def fit(self, X):
        """
        Fits clusters using KMeans algorithm with max_iters iterations
        :param X: (array-like) Feature Vectors
        :return: (list of lists) Points in lists corresponding to centroid
        """

        try:
            self.centroids = self._initialize_centroids(self.k, X)  # Initialize Centroids
            for alg_iter in range(self.max_iter):
                # Sort each datapoint, assigning to nearest centroid
                self.sorted_points = [[] for _ in range(self.k)]  # Each point is placed within one of these k lists

                for x in X:
                    dists = euclidean(x, self.centroids)  # Get distance from x to each
                    centroid_idx = np.argmin(dists)  # Get minimum distance centroid
                    self.sorted_points[centroid_idx].append(x)  # Associate x with that centroid

                prev_centroids = self.centroids
                # Recalculate centroids as means of all associated points
                self.centroids = [np.mean(cluster, axis=0) for cluster in self.sorted_points]

                for i, centroid in enumerate(self.centroids):  # TODO: Why is this how we handle this?
                    if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                        self.centroids[i] = prev_centroids[i]

                return self.sorted_points

        except Exception as e:
            print(f"Error during fitting; exception: {e}")
            raise e


    def evaluate(self, X):
        """
        Returns the index (class) and value of the centroid associated with each point in X
        :param X: Feature vectors
        :return: (list, list)
            - corr_centroid: The centroid values to which each point x is associated
            - classification: The index (class) to which each point x is associated
        """

        corr_centroid = []
        classification = []
        for x in X:
            dists = euclidean(x, self.centroids)  # Get distance of x from each centroid
            centroid_idx = np.argmin(dists)  # Get the centroid index (class) of x
            corr_centroid.append(self.centroids[centroid_idx])  # Add the centroid associated with x to the return structure
            classification.append(centroid_idx)  # Add the assigned class of x to the return structure

        return corr_centroid, classification

    def inertia(self):
        """
        Calculates the WCSS of the model

        :return: (float) The model's WCSS
        """

        wcss = 0
        for i in range(self.k):
            cluster_points = np.array(self.sorted_points[i])
            centroid = self.centroids[i]
            wcss += np.sum((cluster_points - centroid) ** 2)
        return wcss

    def classification_error(self, y):
        """
        Calculates the classification success ratio, if classes of data are known.
        :param y: Actual class labels
        :return: (float) the proportion of correctly classified samples
        """
        return np.sum(self.classification == y) / (len(self.classification))


class DBSCAN:
    """
    Density-based
    """