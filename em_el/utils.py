import numpy as np

__all__=[
    'euclidean',
    'entropy'
]

def euclidean(a, b):
    return np.linalg.norm(a - b)

def entropy(class_probabilities: list) -> float:
    return sum([-p * np.log2(p) for p in class_probabilities if p > 0])