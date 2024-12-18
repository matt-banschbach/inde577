# Neural Networks

## Overview

Neural networks are machine learning models inspired by the structure and function of the human brain. They consist of interconnected nodes, or artificial neurons, organized in layers: an input layer, one or more hidden layers, and an output layer[1]. These networks are designed to recognize patterns, process complex data, and make decisions in a manner similar to biological neurons.

### Mathematical Foundation

The underlying mathematics of neural networks involves:

1. Weighted connections: Each connection between nodes has an associated weight.
2. Activation functions: Nodes use non-linear functions like sigmoid or ReLU to introduce non-linearity.
3. Forward propagation: The input is processed through the network using the formula:

   $$z = \sum_{i=1}^n w_i x_i + b$$
   $$\hat{y} = \sigma(z) $$

   Where $w_i$ are weights, $x_i$ are inputs, $b$ is bias, and $\sigma$ is the activation function.

4. Backpropagation: The network learns by adjusting weights to minimize the difference between predicted and actual outputs.

#### Backpropagation
Backpropagation is a fundamental algorithm used in training neural networks, designed to minimize the error between predicted 
and actual outputs by adjusting the network's weights and biases. The algorithm works by propagating the error backward 
through the network, hence its name.

The core of backpropagation relies on the chain rule from calculus. For a neural network with a loss function L, 
weights w, and inputs x, the goal is to compute ∂L/∂w. This is achieved by breaking down the computation into smaller steps:

$$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w} $$

Where:
- y is the output of a neuron
- z is the weighted sum of inputs to a neuron

The algorithm computes these partial derivatives layer by layer, moving backward from the output.

##### Algorithm Steps

1. Forward Pass:
   - Input data is fed through the network
   - Compute the output of each neuron using the activation function
   - Calculate the loss using a predefined loss function

2. Backward Pass:
   - Compute the gradient of the loss with respect to the output layer
   - Propagate the gradient backward through the network
   - Update weights and biases using an optimization algorithm (e.g., gradient descent)

The update rule for weights is generally:

$$ w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w} $$

Where α is the learning rate[2][5].

##### Efficiency and Implementation

Backpropagation is computationally efficient because it reuses intermediate results from the forward pass. This avoids redundant calculations and makes the algorithm suitable for training large neural networks[7].

The algorithm considers all neurons equally and calculates their derivatives for each backward pass. This allows for parallel computation, making it well-suited for implementation on GPUs[5].
___

### Benefits

1. Handling unstructured data: Neural networks excel at processing large volumes of raw, unorganized data.
2. Continuous learning: They improve accuracy over time through iterative training.
3. Versatility: Neural networks can be applied to various tasks, including image recognition, natural language processing, and speech recognition.
4. Parallel processing: Their structure allows for efficient parallel computation.

### Limitations

1. Black box nature: It's often difficult to interpret how neural networks arrive at their decisions.
2. Data requirements: They typically need large amounts of labeled data for training.
3. Computational intensity: Training neural networks can be computationally expensive and time-consuming.
4. Potential for overfitting: Without proper regularization, they may perform poorly on new, unseen data.
5. Hardware dependency: Complex neural networks often require specialized hardware for efficient training and deployment.

___

### Typical Uses and Purposes

1. Computer vision: Image classification, object detection, and facial recognition[4][8].
2. Natural Language Processing (NLP): Language translation, sentiment analysis, and chatbots[8].
3. Speech recognition: Converting spoken words to text for virtual assistants and transcription services[8].
4. Autonomous vehicles: Analyzing sensor data for real-time decision making[8].
5. Healthcare: Medical image analysis for disease detection and diagnosis[8].
6. Financial forecasting: Predicting market trends and stock prices[13].
7. Recommendation systems: Personalizing content and product suggestions on platforms like Netflix and Instagram[12].

## Directory Contents

Features Neural Network Implementations

**Algorithms Implemented**
- Dense Neural Network (DNN) on MNIST Hand-written number dataset
- Dense Neural Network (DNN) on Wine Dataset


## Datasets

The following datasets are used:
- wine
- MNIST

They can be loaded as follows:

MNIST: 

```python
import keras
(train_X, train_y), (test_X, test_y) = keras.datasets.mnist.load_data()
```

Wine:

```python
from em_el.datasets import load_wine
wine = load_wine()
```

## Reproducibility
Ensure `random_state` parameter is set to 42 for all uses

To use the NPORS Dataset:
- Ensure the pyreadstat package is installed using pip, and imported into the notebook
  - pyreadstat is necessary for Pandas to read SPSS (.sav) files
- Ensure you have downloaded the NPORS dataset and placed the file into the current working directory of the notebook

pyreadstat can be installed using `pip install pyreadstat` and imported using
```python
import pyreadstat
```
