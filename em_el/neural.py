import numpy as np

from em_el.loss_functions import *
from em_el.activations import *
from em_el.utils import *


class DenseNetwork:


    def __init__(self):
        self.layers = [784, 60, 60, 10]
        self.W, self.B = self._initialize_weights()

    @staticmethod
    def _forward_pass(W, B, xi, predict_vector=False):
        Z = [[0.0]]
        A = [xi]
        # L = len(W) - 1

        for i in range(1, len(W)):
            z = W[i] @ A[i - 1] + B[i]
            Z.append(z)

            a = sigmoid(z)
            A.append(a)

        if not predict_vector:
            return Z, A
        else:
            return A[-1]

    @staticmethod
    def _initialize_weights():
        layers = [784, 60, 60, 10]  # Number nodes in each later, from input through output
        W = [[0.0]]
        B = [[0.0]]
        for i in range(1, len(layers)):
            w_temp = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])  # Scaling Factor
            b_temp = np.random.randn(layers[i], 1) * np.sqrt(2 / layers[i - 1])  # Scaling Factor

            W.append(w_temp)
            B.append(b_temp)
        return W, B



    def train(self, X_train, y_train, alpha=0.046, epochs=4):
        # Print the initial mean squared error
        self.cost = [self.model_MSE(self.W, self.B, X_train, y_train)]
        print(f"Starting Cost = {self.cost[0]}")

        # Find the number of non-input layers.
        L = len(self.layers) - 1

        # For each epoch perform stochastic gradient descent.
        for k in range(epochs):
            # Loop over each (xi, yi) training pair of data.
            for xi, yi in zip(X_train, y_train):
                # Use the forward pass function defined before
                # and find the preactivation and post-activation values.
                Z, A = self._forward_pass(self.W, self.B, xi)

                # Store the errors in a dictionary for clear interpretation
                # of computation of these values.
                deltas = dict()

                # Compute the output error
                output_error = (A[L] - yi) * d_sigmoid(Z[L])
                deltas[L] = output_error

                # Loop from L-1 to 1. Recall the right entry of the range function
                # is non-inclusive.
                for i in range(L - 1, 0, -1):
                    # Compute the node errors at each hidden layer
                    deltas[i] = (self.W[i + 1].T @ deltas[i + 1]) * d_sigmoid(Z[i])

                # Loop over each hidden layer and the output layer to perform gradient
                # descent.
                for i in range(1, L + 1):
                    self.W[i] -= alpha * deltas[i] @ A[i - 1].T
                    self.B[i] -= alpha * deltas[i]

            # Show the user the cost over all training examples
            self.cost.append(self.model_MSE(self.W, self.B, X_train, y_train))
            print(f"{k + 1}-Epoch Cost = {self.cost[-1]}")

    def model_MSE(self, W, B, X, y):
        cost = 0.0
        m = 0
        for xi, yi in zip(X, y):
            a = self._forward_pass(W, B, xi, predict_vector=True)
            cost += mse(a, yi)
            m += 1
        return cost / m

    def predict(self, xi):
        # depth = len(self.layers)
        _, A = self._forward_pass(self.W, self.B, xi)
        return np.argmax(A[-1])