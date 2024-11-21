import numpy as np

from em_el.loss_functions import *
from em_el.activations import *
from em_el.utils import *


class DenseNetwork:
    """
    A simple neural network
    """

    def __init__(self, dimensions, activation):
        """
        Initialize a Dense Neural Network

        :param dimensions: (array-like) Including the input and output dimensions, and the dimensions of each hidden layer
        :param activation: (function) Activation function

        """
        self.cost = None
        self.layers = dimensions
        self.W, self.B = self._initialize_weights(dimensions)
        self.activation = activation


    @staticmethod
    def _forward_pass(W, B, xi, predict_vector=False):
        """

        :param W: (array-like) Model Weights
        :param B: (array-like) Model Biases
        :param xi: (array-like) A feature vector
        :param predict_vector: (boolean) Indicator for whether a specific prediction should be made
        :return:
        """
        Z = [[0.0]]  # Pre-activation values
        A = [xi]  # Post activation values
        # L = len(W) - 1

        for i in range(1, len(W)):
            z = W[i] @ A[i - 1] + B[i]  # Calculate pre-activation
            Z.append(z)

            a = sigmoid(z)  # Calculate Activation
            A.append(a)

        if not predict_vector:
            return Z, A
        else:
            return A[-1]


    @staticmethod
    def _initialize_weights(layers):
        """
        Initializes weights for a dense neural network based on supplied dimensions

        :param layers: (array-like) Neural Network dimensions; first element is the input layer and final element is the output layer

        :return: (tuple) W and B ---
            W: (np.ndarray) Initial Weights
            B: (np.ndarray) Initial Biases
        """
        # layers = [784, 60, 60, 10]  # Number nodes in each later, from input through output
        W = [[0.0]]
        B = [[0.0]]
        for i in range(1, len(layers)):
            w_temp = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])  # Scaling Factor
            b_temp = np.random.randn(layers[i], 1) * np.sqrt(2 / layers[i - 1])  # Scaling Factor

            W.append(w_temp)
            B.append(b_temp)
        return W, B


    def fit(self, X_train, y_train, alpha=0.046, epochs=4):
        """
        Fits training data to a dense neural network with layer parameters provided in initialization

        :param X_train: Training feature vectors
        :param y_train: Training labels
        :param alpha: Learning rate
        :param epochs: # of epochs

        :return: self
        """
        # Print the initial mean squared error
        self.cost = [self.model_MSE(self.W, self.B, X_train, y_train)]
        print(f"Starting Cost = {self.cost[0]}")

        # Number of non-input layers.
        L = len(self.layers) - 1

        # Stochastic gradient descent.
        for k in range(epochs):
            # Loop over each (xi, yi) training pair of data.
            for xi, yi in zip(X_train, y_train):
                # Find the preactivation and post-activation values with forward pass
                Z, A = self._forward_pass(self.W, self.B, xi)

                # Compute and store output error
                deltas = {}
                output_error = (A[L] - yi) * d_sigmoid(Z[L])
                deltas[L] = output_error

                for i in range(L - 1, 0, -1):
                    deltas[i] = (self.W[i + 1].T @ deltas[i + 1]) * d_sigmoid(Z[i])  # Compute the node errors at each hidden layer

                # Gradient descent over each hidden layer and output layer
                for i in range(1, L + 1):
                    self.W[i] -= alpha * deltas[i] @ A[i - 1].T
                    self.B[i] -= alpha * deltas[i]

            # Output cost over current epoch
            self.cost.append(self.model_MSE(self.W, self.B, X_train, y_train))
            print(f"{k + 1}-Epoch Cost = {self.cost[-1]}")

        return self


    def model_MSE(self, X, y):
        """
        Calculates the total model Mean Squared Error

        :param X: (array-like) Training feature vectors
        :param y: (array-like) Training labels

        :return: (float) Model Mean Squared Error
        """
        cost = 0.0
        m = 0
        for xi, yi in zip(X, y):
            a = self._forward_pass(self.W, self.B, xi, predict_vector=True)
            cost += mse(a, yi)
            m += 1
        return cost / m


    def predict(self, xi):
        """
        Predicts the label of an input feature vector
        :param xi: (array-like) Feature vector
        :return: The label in onehot encoding
        """
        # depth = len(self.layers)
        _, A = self._forward_pass(self.W, self.B, xi)
        return np.argmax(A[-1])