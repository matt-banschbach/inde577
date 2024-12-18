from em_el.neuron import *
from em_el.utils import *
from em_el.activations import *
from em_el.loss_functions import *
import operator
import numpy as np



class KNN:
    """
    K-Nearest Neighbors
    """

    def __init__(self, K):
        """
        Initialize a new KNN object
        :param K: (int) Hyperparameter K---the number of neighbors
        """
        self.X = None
        self.y = None
        self.K = K

    @staticmethod
    def _k_nearest(target, X, y, K):
        """
        Finds the K nearest neighbors of a given point in a dataset.

        :param X: A numpy array of shape (n_samples, n_features) containing the feature vectors of the training data.
        :param y: A numpy array of shape (n_samples,) containing the corresponding labels for the training data.
        :param x_n: A numpy array of shape (n_features,) representing the query point.
        :param K: The number of nearest neighbors to return.

        :return: A list of tuples, where each tuple contains:
                - The distance to the query point.
                - The feature vector of the neighbor.
                - The label of the neighbor.
            The list is sorted in ascending order of distance.
        """
        distances = [(euclidean(x, target), x, y) for x, y in zip(X, y)]
        distances = sorted(distances, key=operator.itemgetter(0))  # Sort the list by the distances

        return distances[:K]


    def fit(self, X_train, y_train):
        """

        :param X_train: (np.ndarray) A numpy array of Feature vectors
        :param y_train: (np.ndarray) A numpy array of corresponding labels
        :return: self
        """
        self.X = X_train
        self.y = y_train

        return self


    def predict(self, target, regression=False):
        """
        Implements the K-Nearest Neighbors (KNN) algorithm for classification or regression.

        For classification:
            - Finds the K nearest neighbors to the `target` point.
            - Returns the most frequent label among the K nearest neighbors.

        For regression:
            - Finds the K nearest neighbors to the `target` point.
            - Returns the average target value of the K nearest neighbors.

        :param target: A numpy array representing the feature vector of the point to classify or regress.
        :param K: The number of nearest neighbors to consider (default is 3).
        :param regression: A boolean flag indicating whether to perform regression (True) or classification (False).

        :return: predicted label or target value for the input `target` point.
        """
        neighbors = self._k_nearest(target, self.X, self.y, self.K)
        if regression:
            return np.mean([x[2] for x in neighbors])
        else:
            labels = [n[2] for n in neighbors]
            return max(labels, key=labels.count)


    def classification_error(self, test_X, test_y):
        """
        Calculates the classification error rate of a K-Nearest Neighbors (KNN) classifier.

        :param test_X: (np.ndarray) A numpy array containing the feature vectors of the test data.
        :param test_y: (np.ndarray) A numpy array containing the true labels of the test data.

        :return: The classification error rate of the K-Nearest Neighbors (KNN) classifier, which is the proportion of misclassified test points.
        """
        error = 0
        for x_i, y_i in zip(test_X, test_y):
            error += y_i != self.predict(self, x_i)
        return error / len(test_X)


class PerceptronClassifier:
    """
    Simple, Single Neuron Perceptron Classifier
    """

    def __init__(self):
        self.neuron = SingleNeuron(sign, mse)

    def train(self, X, y, alpha, epochs):
        """
        Fits Logistic Regression model to features X and labels y

        :param X: Feature array
        :param y: Label Vector
        :param alpha: Learning Rate
        :param epochs: # of Epochs
        """

        self.neuron.train(X, y, alpha, epochs)
        self.model_errors = self.neuron.model_errors

    def predict(self, x):
        """
        Predicts label y for feature entry x

        :param x: A feature entry
        :return: A predicted label based on feature entry x and model weights
        """

        return self.neuron.predict(x)