# Unsupervised Learning

This directory features several unsupervised machine learning algorithms:
- K-Means Clustering
- DBSCAN
- PCA
- Image Compression with SVD

### Unsupervised Machine Learning Overview
Unsupervised machine learning is a type of AI technique that identifies patterns and relationships in unlabeled data 
without explicit guidance. It operates by analyzing raw datasets to discover hidden structures, 
groupings, and insights autonomously.

From a mathematical perspective, unsupervised learning algorithms aim to determine a function from unlabeled data 
without error or reward signals. Key approaches include:

1. Clustering: Grouping similar data points together
2. Dimensionality reduction: Reducing the number of features while preserving important information
3. Association: Finding relationships between variables

One statistical method used is the method of moments, which estimates probability distribution parameters using the 
expected values of powers of the parameters.

## Benefits and Limitations

Benefits of unsupervised learning include:

- Discovering hidden patterns and insights in data without human supervision
- Ability to work with large amounts of unlabeled data
- Useful for exploratory data analysis and initial data understanding

Limitations include:

- Lack of precision in outcome interpretation due to absence of labeled data
- High dependency on data quality, as biases or anomalies can lead to misleading results
- Results can be subjective and prone to overfitting

## Integration with Supervised Learning

Unsupervised and supervised learning are often used together in machine learning pipelines:

1. Unsupervised learning can be used for initial data exploration and feature extraction before applying supervised methods.
2. It can help reduce the dimensionality of data, making subsequent supervised learning more efficient.
3. Unsupervised techniques like clustering can create pseudo-labels for semi-supervised learning approaches.

By combining these approaches, practitioners can leverage the strengths of both methods to build more robust and 
effective machine learning models.
___
### Directory Contents

**Clustering**
- K-Means (from scratch)
- DBSCAN (from scratch)

**Dimensionality Reduction**
- PCA (sklearn)

**Image Compression**
- Image Compression with SVD