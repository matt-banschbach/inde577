import matplotlib.pyplot as plt
import numpy as np
from em_el.utils import *


__all__ = ['SingleNeuron']

class SingleNeuron(object):
    """
    A class used to represent a single artificial neuron.

    :param activation_function: function
        The activation function applied to the preactivation linear combination.

    :param cost_function: function
        The cost function used to measure model performance.

    Attributes
    --------

    activation_function: function
        The activation function applied to the preactivation linear combination.

    cost_function: function
        The cost function used to measure model performance.

    weight: numpy.ndarray
        The weights and bias of the single neuron. The last entry being the bias.
        This attribute is created when the train method is called.

    errors_: list
        A list containing the mean sqaured error computed after each iteration
        of stochastic gradient descent per epoch.

    Methods
    -------
    train(self, X, y, alpha = 0.005, epochs = 50)
        Iterates the stochastic gradient descent algorithm through each sample
        a total of epochs number of times with learning rate alpha. The data
        used consists of feature vectors X and associated labels y.

    predict(self, X)
        Uses the weights and bias, the feature vectors in X, and the
        activation_function to make a y_hat prediction on each feature vector.
    """

    def __init__(self, activation_function, cost_function):
        """

        :param activation_function: activation function
        :param cost_function: cost function
        """
        self.activation_function = activation_function
        self.cost_function = cost_function

    def train(self, X, y, alpha = 0.005, epochs = 50):
        """
        Fits the neuron to feature vectors X and labels y.

        :param X: (np.ndarray) Feature vectors
        :param y: (np.ndarray) Labels
        :param alpha: (float) learning rate
        :param epochs: (int) number of epochs
        :return: The neuron object
        """

        self.weight = np.random.rand(X.shape[1])
        self.bias = np.random.rand()
        self.model_errors = []

        N = X.shape[0]

        for _ in range(epochs):
            epoch_error = 0
            for xi, yi in zip(X, y):
                entry_error = (self.predict(xi) - yi)
                self.weight -= alpha * entry_error * xi
                self.bias -= alpha * entry_error
                epoch_error += self.cost_function(self.predict(xi), yi)
            self.model_errors.append(epoch_error / N)
        return self

    def predict(self, x):
        """
        Based on the supplied activation function, predicts a

        :param x: (np.ndarray) feature entry
        :return: A predicted value based on the feature entry and model weights
        """
        preactivation = np.dot(x, self.weight) + self.bias
        return self.activation_function(preactivation)

    def plot_cost_function(self):
        """
        Plots the epoch-wise cost based on the cost function supplied during model initialization
        """

        fig, axs = plt.subplots(figsize = (10, 8))
        axs.plot(range(1, len(self.model_errors) + 1), self.model_errors, label ="Cost function")
        axs.set_xlabel("epochs", fontsize = 15)
        axs.set_ylabel("Cost", fontsize = 15)
        axs.legend(fontsize = 15)
        axs.set_title("Cost Calculated after Epoch During Training", fontsize = 18)
        plt.show()