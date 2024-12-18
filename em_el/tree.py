import numpy as np
from collections import Counter
from em_el.utils import entropy  # Ensure this import is correct

class TreeNode:
    """
    A class representing a node in the decision tree.

    Attributes:
        data (np.array): The data associated with the node.
        feature_idx (int): The index of the feature used for splitting.
        feature_val (float): The value of the feature used for splitting.
        prediction_probs (np.array): The prediction probabilities for the node.
        information_gain (float): The information gain of the split.
        feature_importance (float): The importance of the feature used for splitting.
        left (TreeNode): The left child node.
        right (TreeNode): The right child node.
    """

    def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain):
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.feature_importance = self.data.shape[0] * self.information_gain
        self.left = None
        self.right = None

    def node_def(self) -> str:
        """
        Returns a string representation of the node.

        Returns:
            str: A string describing the node.
        """
        if self.left or self.right:
            return (f"NODE | Information Gain = {self.information_gain} | "
                    f"Split IF X[{self.feature_idx}] < {self.feature_val} THEN left O/W right")
        else:
            unique_values, value_counts = np.unique(self.data[:, -1], return_counts=True)
            output = ", ".join([f"{value}->{count}" for value, count in zip(unique_values, value_counts)])
            return f"LEAF | Label Counts = {output} | Pred Probs = {self.prediction_probs}"

class DecisionTree:
    """
    A class representing a decision tree classifier.

    Attributes:
        max_depth (int): The maximum depth of the tree.
        min_samples_leaf (int): The minimum number of samples required to be in a leaf.
        min_information_gain (float): The minimum information gain required to make a split.
        numb_of_features_splitting (str): The method for selecting features to split on.
        amount_of_say (float): A parameter used for the Adaboost algorithm.
    """

    def __init__(self, max_depth=4, min_samples_leaf=1,
                 min_information_gain=0.0, numb_of_features_splitting=None,
                 amount_of_say=None) -> None:
        """
        Initializes the decision tree with the given hyperparameters.

        Args:
            max_depth (int): The maximum depth of the tree.
            min_samples_leaf (int): The minimum number of samples required to be in a leaf.
            min_information_gain (float): The minimum information gain required to make a split.
            numb_of_features_splitting (str): The method for selecting features to split on.
            amount_of_say (float): A parameter used for the Adaboost algorithm.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.amount_of_say = amount_of_say

    @staticmethod
    def _class_probabilities(labels: list) -> list:
        """
        Calculates the class probabilities for a list of labels.

        Args:
            labels (list): A list of labels.

        Returns:
            list: A list of class probabilities.
        """
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]

    def _data_entropy(self, labels: list) -> float:
        """
        Calculates the entropy of a list of labels.

        Args:
            labels (list): A list of labels.

        Returns:
            float: The entropy of the labels.
        """
        return entropy(self._class_probabilities(labels))

    def _partition_entropy(self, subsets: list) -> float:
        """
        Calculates the partition entropy for a list of subsets.

        Args:
            subsets (list): A list of label lists.

        Returns:
            float: The partition entropy.
        """
        total_count = sum([len(subset) for subset in subsets])
        return sum([self._data_entropy(subset) * (len(subset) / total_count) for subset in subsets])

    @staticmethod
    def _split(data: np.array, feature_idx: int, feature_val: float) -> tuple:
        """
        Splits the data based on a feature index and value.

        Args:
            data (np.array): The data to split.
            feature_idx (int): The index of the feature to split on.
            feature_val (float): The value of the feature to split on.

        Returns:
            tuple: Two groups of data split based on the feature value.
        """
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]
        return group1, group2

    def _select_features_to_use(self, data: np.array) -> list:
        """
        Selects the features to use for splitting based on the hyperparameter.

        Args:
            data (np.array): The data to select features from.

        Returns:
            list: A list of feature indices to use for splitting.
        """
        feature_idx = list(range(data.shape[1] - 1))
        if self.numb_of_features_splitting == "sqrt":
            feature_idx_to_use = np.random.choice(feature_idx, size=int(np.sqrt(len(feature_idx))))
        elif self.numb_of_features_splitting == "log":
            feature_idx_to_use = np.random.choice(feature_idx, size=int(np.log2(len(feature_idx))))
        else:
            feature_idx_to_use = feature_idx
        return feature_idx_to_use

    def _find_best_split(self, data: np.array) -> tuple:
        """
        Finds the best split for the data based on entropy.

        Args:
            data (np.array): The data to find the best split for.

        Returns:
            tuple: The two split groups, the feature index, the feature value, and the partition entropy.
        """
        min_part_entropy = 1e9
        feature_idx_to_use = self._select_features_to_use(data)
        for idx in feature_idx_to_use:
            feature_vals = np.percentile(data[:, idx], q=np.arange(25, 100, 25))
            for feature_val in feature_vals:
                g1, g2 = self._split(data, idx, feature_val)
                part_entropy = self._partition_entropy([g1[:, -1], g2[:, -1]])
                if part_entropy < min_part_entropy:
                    min_part_entropy = part_entropy
                    min_entropy_feature_idx = idx
                    min_entropy_feature_val = feature_val
                    g1_min, g2_min = g1, g2
        return g1_min, g2_min, min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy

    def _find_label_probs(self, data: np.array) -> np.array:
        """
        Calculates the label probabilities for the data.

        Args:
            data (np.array): The data to calculate label probabilities for.

        Returns:
            np.array: An array of label probabilities.
        """
        labels_as_integers = data[:, -1].astype(int)
        total_labels = len(labels_as_integers)
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels
        return label_probabilities

    def _create_tree(self, data: np.array, current_depth: int) -> TreeNode:
        """
        Recursively creates the decision tree.

        Args:
            data (np.array): The data to create the tree from.
            current_depth (int): The current depth of the tree.

        Returns:
            TreeNode: The root node of the created tree.
        """
        if current_depth > self.max_depth:
            return None
        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self._find_best_split(data)
        label_probabilities = self._find_label_probs(data)
        node_entropy = entropy(label_probabilities)
        information_gain = node_entropy - split_entropy
        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node
        elif information_gain < self.min_information_gain:
            return node
        current_depth += 1
        node.left = self._create_tree(split_1_data, current_depth)
        node.right = self._create_tree(split_2_data, current_depth)
        return node

    def _predict_one_sample(self, X: np.array) -> np.array:
        """
        Predicts the label probabilities for a single sample.

        Args:
            X (np.array): The sample to predict for.

        Returns:
            np.array: The predicted label probabilities.
        """
        node = self.tree
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right
        return pred_probs

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """
        Trains the decision tree with the given training data.

        Args:
            X_train (np.array): The training features.
            Y_train (np.array): The training labels.
        """
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)
        self.tree = self._create_tree(data=train_data, current_depth=0)
        self.feature_importances = dict.fromkeys(range(X_train.shape[1]), 0)
        self._calculate_feature_importance(self.tree)
        total_importance = sum(self.feature_importances.values())
        self.feature_importances = {k: v / total_importance for k, v in self.feature_importances.items()}

    def predict_proba(self, X_set: np.array) -> np.array:
        """
        Predicts the label probabilities for a set of samples.

        Args:
            X_set (np.array): The set of samples to predict for.

        Returns:
            np.array: The predicted label probabilities for the set of samples.
        """
        pred_probs = np.apply_along_axis(self._predict_one_sample, 1, X_set)
        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """
        Predicts the labels for a set of samples.

        Args:
            X_set (np.array): The set of samples to predict for.

        Returns:
            np.array: The predicted labels for the set of samples.
        """
        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        return preds

    def _print_recursive(self, node: TreeNode, level=0) -> None:
        """
        Recursively prints the decision tree.

        Args:
            node (TreeNode): The current node to print.
            level (int): The current level of the tree.
        """
        if node is not None:
            self._print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' + node.node_def())
            self._print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        """
        Prints the decision tree.
        """
        self._print_recursive(node=self.tree)

    def _calculate_feature_importance(self, node):
        """
        Recursively calculates the feature importance by visiting each node in the tree.

        Args:
            node (TreeNode): The current node to calculate feature importance for.
        """
        if node is not None:
            self.feature_importances[node.feature_idx] += node.feature_importance
            self._calculate_feature_importance(node.left)
            self._calculate_feature_importance(node.right)
