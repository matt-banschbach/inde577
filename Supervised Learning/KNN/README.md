# K-Nearest Neighbors (KNN)
___

### Description
# K-nearest neighbors (KNN)
___

The K-nearest neighbors (KNN) algorithm is a supervised machine learning algorithm used for classification and regression tasks. It works by finding the K closest data points to a query point and making predictions based on their labels or values.

#### Mathematical Foundation

The KNN algorithm relies on distance metrics to determine similarity between data points. The most common distance metric used is Euclidean distance, calculated as:

$$d = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

where x and y are two data points with n features.

#### Purpose and Functionality

KNN classifies new data points based on the majority class of their K nearest neighbors. The value of K is a hyperparameter that determines how many neighbors to consider. For classification tasks, KNN uses majority voting, while for regression, it uses the average of the K neighbors' values.

#### Benefits

1. Simple implementation and intuitive concept
2. No training phase required, making it time-efficient for quick modeling
3. Easily adapts to multi-class problems
4. Non-parametric, making no assumptions about data distribution

#### Limitations

1. Computationally expensive for large datasets, especially during prediction
2. Sensitive to the curse of dimensionality, performing poorly with high-dimensional data
3. Requires careful selection of K and distance metric
4. Prone to overfitting with small K values and underfitting with large K values
5. Lack of interpretability in predictions

KNN's performance depends heavily on the choice of K and the distance metric used. Cross-validation can help determine the optimal K value for a given dataset. Despite its limitations, KNN remains a powerful and versatile algorithm for both classification and regression tasks due to its simplicity and effectiveness.
___

### Datasets

I implement KNN on several basic datasets from Seaborn in `knn_basic.ipynb`

**Seaborn Datasets**
- penguins
- wine

to use these datasets, ensure that the seaborn package is installed. Data can then be loaded using `sns.load_dataset()`. 
For example, the penguins dataset can be loaded as such:

```python
import seaborn as sns
sns.load_dataset('penguins')
```
