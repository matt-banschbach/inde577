# Linear Model
___

This directory contains implementations of Linear ML Algorithms

**Algorithms**:
1. Percpetron
2. Linear Regression
3. Logistic Regression

(See each notebook for an overview of each algorithm)


### Datasets

The following datasets are used
- penguins
- wine

And can be loaded as follows:

Penguins: 

```python
import seaborn as sns
sns.load_dataset('penguins')
```

Wine:

```python
from em_el.datasets import load_wine
wine = load_wine()
```

### Reproducibility
Ensure `random_state` parameter is set to 42 for all uses
