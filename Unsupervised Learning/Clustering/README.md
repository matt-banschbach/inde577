# Clustering

This directory implements two unsupervised clustering algorithms:
1. K-Means
2. DBSCAN

### K-Means Overview

K-Means clustering is a popular unsupervised learning algorithm used to partition data into K distinct, non-overlapping clusters. It aims to minimize the intra-cluster distances while maximizing inter-cluster distances.

##### Underlying Mathematics

1. Euclidean Distance: Used to calculate the distance between data points and centroids:

   $$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

2. Centroid Calculation: The mean of all points in a cluster:

   $$c_k = \frac{1}{n_k} \sum_{i=1}^{n_k} x_i$$

   where $c_k$ is the centroid of cluster k, and $n_k$ is the number of points in that cluster.

3. Objective Function: Minimizing the sum of squared distances between points and their assigned centroids:

   $$J = \sum_{k=1}^K \sum_{i=1}^{n_k} \|x_i - c_k\|^2$$

##### Benefits

1. Simplicity and Efficiency: Easy to implement and computationally efficient, especially for large datasets.
2. Scalability: Easily scalable to large datasets and generalizes to clusters of different shapes and sizes.
3. Interpretability: Results are easy to interpret and explain.
4. Guaranteed Convergence: The algorithm is guaranteed to converge, although it may not always find the global optimum.
5. Flexibility: Can be adapted for various types of data and applications.

##### Limitations

1. Sensitivity to Initial Conditions: Results can vary based on the initial random selection of centroids.

2. Predefined K: Requires the number of clusters (K) to be specified in advance, which can be challenging to determine.

3. Sensitivity to Outliers: Outliers can significantly affect the cluster centroids and overall results.

4. Assumption of Spherical Clusters: Performs poorly when clusters have non-spherical shapes or varying sizes.

5. Local Optima: May converge to local optima instead of the global optimum.

6. Limited to Linear Boundaries: Cannot handle non-linear cluster boundaries effectively.

7. Categorical Data: Struggles with categorical data, as it primarily works with numerical data.

K-Means clustering remains a fundamental algorithm in unsupervised learning due to its simplicity and efficiency.
However, practitioners should be aware of its limitations and consider alternatives or modifications when dealing with
complex datasets or specific clustering requirements.


### DBSCAN Overview

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised machine learning algorithm used for clustering data points based on their density in a feature space.

##### Underlying Mathematics

1. Epsilon (ε): The maximum distance between two points to be considered neighbors.
2. MinPts: The minimum number of points required to form a dense region.
3. Core Points: Points with at least MinPts neighbors within ε distance.
4. Border Points: Points within ε distance of a core point but with fewer than MinPts neighbors.
5. Noise Points: Points that are neither core nor border points.
6. Euclidean Distance: Used to calculate the distance between points:
   $$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$
7. Density: Calculated by counting the number of points within ε distance of a point.
8. Reachability Distance:
   
9. $$\text{reachability\_distance}(p, q) = \max(d(p, q), \text{core\_dist}(q))$$

   where $d(p, q)$ is the Euclidean distance between points $p$ and $q$, and $core_dist(q)$ is the distance between 
10. point $q$ and its closest core point.

##### Benefits

1. Automatically discovers the number of clusters.
2. Effective at identifying and removing noise in a dataset.
3. Can find clusters of arbitrary shape.
4. Does not require the number of clusters to be specified in advance.
5. Works well with non-spherical clusters.
6. More robust than algorithms like K-means for certain types of data.

##### Limitations

1. Sensitive to choice of hyperparameters (ε and MinPts), which may require trial and error.
2. Struggles with clusters of significantly different densities.
3. Performance can be limited on high-dimensional data due to the curse of dimensionality.
4. Assumes local density of data points is somewhat globally uniform.
5. May have difficulty with datasets containing large differences in densities.
6. Choosing a meaningful distance threshold ε can be challenging if the data and scale are not well understood.
7. Not entirely deterministic: border points reachable from multiple clusters can be assigned to either, 
depending on the order of data processing.

DBSCAN is particularly useful for spatial data clustering, outlier detection, and when dealing with datasets where the
number of clusters is unknown. Its ability to handle noise and find arbitrarily shaped clusters makes it valuable in
various fields, including image processing, geospatial analysis, and anomaly detection.


### Datasets

The following datasets are used
- sklearn make_blobs
- sklearn make_moons

and can be imported as follows:

```python
from sklearn.datasets import make_blobs, make_moons

blobs = make_blobs(n_samples=100, n_features=3, random_state=42)
moons = make_moons(n_samples=100, random_state=42)
```

### Reproducibility

Ensure all `random_state` parameters are set to 42