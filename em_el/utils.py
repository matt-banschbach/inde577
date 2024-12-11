import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

__all__=[
    'euclidean',
    'entropy'
]

def euclidean(a, b):
    return np.linalg.norm(a - b)

def entropy(class_probabilities: list) -> float:
    return sum([-p * np.log2(p) for p in class_probabilities if p > 0])


def draw_confusion_matrix(cf_matrix, title):
    plt.figure(figsize=(9, 6))
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', cbar=False)

    ax.set_title(title)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')

    num_rows, num_cols = cf_matrix.shape  # Get the number of rows and columns

    # Draw horizontal lines
    for i in range(num_rows + 1):
        ax.axhline(y=i, color='black', linewidth=0.5)

    # Draw vertical lines
    for j in range(num_cols + 1):
        ax.axvline(x=j, color='black', linewidth=0.5)

    plt.show()