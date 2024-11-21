from em_el.neuron import *
from em_el.utils import *
from em_el.activations import *
from em_el.loss_functions import *




class LinearRegression:
    """
    Single Neuron Linear Regression
    """

    def __init__(self):
        self.neuron = SingleNeuron(constant, mse)


    def train(self, X, y, alpha, epochs):
        """
        Fits Linear Regression model to features X and labels y

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



class LogisticRegression:
    """
    Single Neuron Model for Logistic Regression
    """

    def __init__(self):
        self.neuron = SingleNeuron(sigmoid, binary_cross_entropy)


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