# Decision Trees
___
### Overview

Decision trees are a popular supervised learning algorithm used for both classification and regression tasks in machine learning[1][4]. They create a tree-like structure of decisions based on input features to predict an outcome or classify data.

##### Underlying Math and Information Theory

Decision trees use concepts from information theory to determine the best way to split data:

1. Entropy: A measure of impurity or uncertainty in a dataset[6]. It's calculated as:

   $$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

   where $p_i$ is the proportion of samples belonging to class $i$ in set $S$.

2. Information Gain: The reduction in entropy after a dataset is split on an attribute. It's calculated as:

   $$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

   where $S$ is the dataset, $A$ is the attribute, and $S_v$ is the subset of $S$ where attribute $A$ has value $v$.

The algorithm selects the attribute with the highest information gain as the splitting criterion at each node.

##### Benefits

1. Interpretability: Decision trees provide a clear, visual representation of the decision-making process.
2. Handling various data types: They can work with both numerical and categorical data.
3. Minimal data preparation: Decision trees require less data preprocessing compared to other algorithms.
4. Non-parametric: They make no assumptions about the underlying data distribution.

##### Limitations

1. Overfitting: Decision trees can create overly complex models that don't generalize well to new data.
2. Instability: Small changes in the data can lead to significantly different tree structures.
3. Bias towards dominant classes: In imbalanced datasets, trees may favor the majority class.
4. Lack of smoothness: They create discrete decision boundaries, which can be problematic for some applications.
5. Limited expressiveness: A single tree might not capture complex relationships in the data.

##### Typical Uses and Purpose

Decision trees are commonly used for:
1. Credit Scoring
2. Customer Churn Prediction
3. Medical Diagnosis
4. Fraud Detection
5. Marketing Campaign Optimization
6. Equipment Failure Prediction
7. Species Identification in Biology
8. Sentiment Analysis


### Datasets

The following datasets are used:
- wine
- penguins

And can be loaded as follows:

Wine:
```python
from em_el.datasets import load_wine
wine = load_wine()
```

Penguins:
```python
import seaborn as sns
penguins = sns.load_dataset('penguins')
```

### Reproducibility
Ensure all `random_state` parameters are set to 42