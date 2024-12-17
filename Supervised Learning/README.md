# Supervised Learning

### Overview

Supervised machine learning is a subcategory of machine learning that uses labeled datasets to train algorithms to predict outcomes and recognize patterns[1][9]. The goal is to train a model of the form y = f(x), where y represents the output and x represents the input[8].

##### Mathematical Foundation

The statistical framework for supervised learning involves:

1. Input data: A finite sequence $S = ((x_1, y_1), ..., (x_n, y_n))$ of pairs, where y_i is the label corresponding to x_i.
2. Output: A function h: X → Y, called a hypothesis, that aims to predict y ∈ Y for arbitrary x ∈ X.
3. Learning algorithm: A map A that produces the hypothesis based on the training data.

The learning process involves minimizing a loss function to adjust the model's parameters. For example, in 
linear regression, the Widrow-Hoff algorithm (least-mean-square) is used:

$$ θ := θ + α(Y - Xθ)^T X $$

where $\theta$ represents the model parameters, $\alpha$ is the learning rate, $Y$ is the observed output, and $X$ is the input data.

##### Types of Supervised Learning

1. Classification: Predicts categorical outputs (e.g., spam detection, image classification).
2. Regression: Predicts continuous values (e.g., sales forecasting, weather prediction).

##### Benefits

1. High accuracy: Models can make very accurate predictions on new unseen data when trained on sufficient labeled data.
2. Wide range of algorithms: Many mature algorithms are available, such as linear regression, random forest, SVM, and neural networks.
3. Less prone to overfitting: Labeled data helps reduce the tendency to overfit on training data.
4. Solves real-world problems: Effective for tasks like spam detection, object identification, and image recognition.
5. Reusability: Training data can be reused unless there are feature changes.

##### Limitations

1. Requires large labeled datasets: Creating labeled training data is expensive and time-consuming.
2. Computational intensity: Training time can be significant, especially for complex models.
3. Limited to known patterns: Cannot discover new patterns or classes not present in the training data.
4. Regular updates needed: Models often require frequent updates to maintain accuracy.
5. Preprocessing challenges: Preparing data for prediction can be difficult.
6. Potential for overfitting: If not carefully managed, models can overfit to the training data.
7. Inability to handle complex tasks: Some intricate machine learning tasks may be beyond the capabilities of 
8. supervised learning.


### Directory Contents

##### Linear Model

Features Linear ML Algorithms

**Implementations**
- Perceptron on Penguins
- LinearRegression on Penguins
- LogisticRegression on Penguins


##### Neural Networks

- Dense Neural Network on MNIST Hand-written numbers dataset (written from scratch)
- Dense Neural Network on Wine Dataset
- Dense Neural Network on NPORS Dataset

##### K-Nearest Neighbors (KNN)

- 
- KNN on Penguins Dataset
- 


##### Decision Trees
- 
- Decision Trees

##### Random Forest and Ensemble Methods


___
### Datasets

I primarily use the penguins and wine datasets, which can be loaded with the following

Penguins:
```python
import seaborn as sns
sns.load_dataset('penguins')
```

Wine (using sklearn):

```python
from sklearn.datasets import load_wine
wine = load_wine()
```

OR using my package, em_el:
```python
from em_el.datasets import load_wine
wine = load_wine()
```

the latter method abstracts away necessary preprocessing.