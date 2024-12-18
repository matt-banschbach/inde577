# Random Forests and Ensemble Methods

The following algorithms are implemented:
1. Bootstrap-aggregating
2. Random Forest
3. Voting
   4. Using Random Forest and Support Vector Machine
5. Boosting (adaBoost)

See each notebook for a description of each

### General Overview of Ensemble Methods

Ensemble methods in supervised machine learning are techniques that combine multiple individual models to create a 
more powerful and accurate predictive model. The fundamental principle behind ensemble methods is that a group of 
diverse models can collectively outperform any single model.

The general principle of ensemble methods involves:

1. Creating multiple base models: These are often referred to as "weak learners" or "base learners".

2. Introducing diversity: Ensuring that the base models are different from each other, either through 
varying the training data, using different algorithms, or adjusting model parameters.

3. Aggregating predictions: Combining the outputs of the individual models to produce a final prediction.

The rationale behind ensemble methods is that by leveraging the strengths of multiple models and mitigating 
their individual weaknesses, the ensemble can achieve higher accuracy and robustness. 
This approach helps to reduce overfitting, decrease variance, and improve generalization to unseen data.

Ensemble methods can be broadly categorized into parallel and sequential techniques:

1. Parallel methods: Base learners are trained independently of each other, allowing for simultaneous training.
2. Sequential methods: Each subsequent model is trained to address the errors of the previous models.

The combination of model predictions can be done through various techniques, such as voting 
(for classification tasks) or averaging (for regression tasks). More sophisticated methods may use 
weighted combinations or meta-models to determine the optimal way to combine the base model predictions.

Ensemble methods have proven to be highly effective in many machine learning competitions and real-world applications, 
often producing more accurate and reliable results than individual models alone.
___

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
sns.load_dataset('penguins')
```
___

### Reproducibility

Ensure all `random_state` parameter values are set to 42