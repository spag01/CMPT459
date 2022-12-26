from cmath import inf
from posixpath import split
from typing import Optional, Sequence, Mapping
import numpy as np
import pandas as pd
import random


class Node(object):
    def __init__(self, node_size: int, node_class: str, depth: int, single_class:bool = False):
        # Every node is a leaf unless you set its 'children'
        self.is_leaf = True
        # Each 'decision node' has a name. It should be the feature name
        self.name = None
        # All children of a 'decision node'. Note that only decision nodes have children
        self.children = {}
        # Whether corresponding feature of this node is numerical or not. Only for decision nodes.
        self.is_numerical = None
        # Threshold value for numerical decision nodes. If the value of a specific data is greater than this threshold,
        # it falls under the 'ge' child. Other than that it goes under 'l'. Please check the implementation of
        # get_child_node for a better understanding.
        self.threshold = None
        # The class of a node. It determines the class of the data in this node. In this assignment it should be set as
        # the mode of the classes of data in this node.
        self.node_class = node_class
        # Number of data samples in this node
        self.size = node_size
        # Depth of a node
        self.depth = depth
        # Boolean variable indicating if all the data of this node belongs to only one class. This is condition that you
        # want to be aware of so you stop expanding the tree.
        self.single_class = single_class

    def set_children(self, children):
        self.is_leaf = False
        self.children = children

    def get_child_node(self, feature_value)-> 'Node':
        if not self.is_numerical:
            return self.children[feature_value]
        else:
            if feature_value >= self.threshold:
                return self.children['ge'] # ge stands for greater equal
            else:
                return self.children['l'] # l stands for less than

    def __str__(self):
        return f"Name: {self.name}, Leaf: {self.is_leaf}, Numeric: {self.is_numerical}, Thres:{self.threshold}, Class:{self.node_class}, Size:{self.size}, Depth:{self.depth}, Single:{self.single_class}"

    def print_tree(self):
        print("\t"*self.depth + str(self))
        for k in self.children:
            self.children[k].print_tree()


class RandomForest(object):
    def __init__(self, n_classifiers: int,
                 criterion: Optional['str'] = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = None,
                 max_features: Optional[int] = None):
        """
        :param n_classifiers:
            number of trees to generated in the forrest
        :param criterion:
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini
            impurity and “entropy” for the information gain.
        :param max_depth:
            The maximum depth of the trees.
        :param min_samples_split:
            The minimum number of samples required to be at a leaf node
        :param max_features:
            The number of features to consider for each tree.
        """
        self.n_classifiers = n_classifiers
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.criterion_func = self.entropy if criterion == 'entropy' else self.gini


    def fit(self, X: pd.DataFrame, y_col: str)->float:
        """
        :param X: data
        :param y_col: label column in X
        :return: accuracy of training dataset
        """
        features = self.process_features(X, y_col)
        # Your code
            
        if self.max_features > len(features):
            self.max_features = len(features)
        
        # self.trees.append(self.generate_tree(X, y_col, features))
        for i in range(self.n_classifiers):
            random_X = X.sample(X.shape[0], replace=True)
            random_features = random.sample(features, self.max_features)
            tree = self.generate_tree(random_X, y_col, random_features)
            self.trees.append(tree)

        return self.evaluate(X, y_col)

    def predict(self, X: pd.DataFrame)->np.ndarray:
        """
        :param X: data
        :return: aggregated predictions of all trees on X. Use voting mechanism for aggregation.
        """
        predictions = []
        # Your code
        for index, row in X.iterrows():
            votes = []
            for node in self.trees:
                while not node.is_leaf:
                    feature_value = row[node.name]
                    if node.is_numerical or feature_value in node.children:
                        node = node.get_child_node(feature_value)
                    else:
                        break
                votes.append(node.node_class)
                
            predictions.append(max(set(votes), key=votes.count))

        return np.array(predictions)

    def evaluate(self, X: pd.DataFrame, y_col: str)-> int:
        """
        :param X: data
        :param y_col: label column in X
        :return: accuracy of predictions on X
        """
        preds = self.predict(X)
        acc = sum(preds == X[y_col]) / len(preds)
        return acc

    def generate_tree(self, X: pd.DataFrame, y_col: str,   features: Sequence[Mapping])->Node:
        """
        Method to generate a decision tree. This method uses self.split_tree() method to split a node.
        :param X:
        :param y_col:
        :param features:
        :return: root of the tree
        """
        root = Node(X.shape[0], X[y_col].mode().iloc[0], 0)
        # Your code
        self.split_node(root, X, y_col, features.copy())
        # root.print_tree()
        return root

    def split_node(self, node: Node, X: pd.DataFrame, y_col:str, features: Sequence[Mapping]) -> None:
        """
        This is probably the most important function you will implement. This function takes a node, uses criterion to
        find the best feature to slit it, and splits it into child nodes. I recommend to use revursive programming to
        implement this function but you are of course free to take any programming approach you want to implement it.
        :param node:
        :param X:
        :param y_col:
        :param features:
        :return:
        """

        if X[y_col].unique().shape[0] <= 1:
            node.single_class = True
            return
        
        if node.depth >= self.max_depth:
            return

        if node.size < self.min_samples_split:
            return
        
        max_gain = -float('inf')
        max_gain_feature = None
        for feature in features:
            score_node = self.criterion_func(X, feature, y_col, False)
            score_split = self.criterion_func(X, feature, y_col)
            gain = score_node - score_split
            if gain > max_gain:
                max_gain = gain
                max_gain_feature = feature

        if max_gain_feature is None:
            return
        
        node.name = max_gain_feature['name']
        node.is_numerical = np.issubdtype(max_gain_feature['dtype'], np.number)

        splits = self.split_on_feature(X, max_gain_feature, y_col)

        children = {}
        if node.is_numerical:
            node.threshold = X[node.name].mean().item()
            X_ge = splits[0]
            X_l = splits[1]
            X_ge_mode = X_ge[y_col].mode().iloc[0] if X_ge.shape[0] > 0 else node.node_class
            X_l_mode = X_l[y_col].mode().iloc[0] if X_l.shape[0] > 0 else node.node_class
            child_ge = Node(X_ge.shape[0], X_ge_mode, node.depth + 1)
            child_l = Node(X_l.shape[0], X_l_mode, node.depth + 1)

            self.split_node(child_ge, X_ge, y_col, features)
            self.split_node(child_l, X_l, y_col, features)

            children['ge'] = child_ge
            children['l'] = child_l

        else:
            new_features = features.copy()
            new_features.remove(feature)
            for split in splits:
                child = Node(split.shape[0], split[y_col].mode().iloc[0], node.depth + 1)
                self.split_node(child, split, y_col, new_features)
                children[split[node.name].iloc[0]] = child

        node.set_children(children)


    def split_on_feature(self, X: pd.DataFrame, feature: Mapping, y_col: str) -> Sequence[pd.Series]:
        """
        Returns gini index of the give feature
        :param X: data
        :param feature: the feature you want to use to get compute gini score
        :param y_col: name of the label column in X
        :return:
        """
        name = feature['name']
        if np.issubdtype(feature['dtype'], np.number):
            threshold = X[name].mean().item()
            
            ge = X[X[name] >= threshold]
            l = X[X[name] < threshold]

            return [ge, l]

        else:
            gb = X.groupby(name)
            return [gb.get_group(g) for g in gb.groups]


    def gini(self, X: pd.DataFrame, feature: Mapping, y_col: str, split: bool = True) -> float:
        """
        Returns gini index of the give feature
        :param X: data
        :param feature: the feature you want to use to get compute gini score
        :param y_col: name of the label column in X
        :return:
        """
        def split_gini(y: pd.Series):
            ps = y.value_counts(normalize=True)
            return 1 - (ps**2).sum()

        len = X.shape[0]
        splits = self.split_on_feature(X, feature, y_col) if split else [X]

        res = 0
        for split in splits:
            en = split_gini(split[y_col])
            len_g = split.shape[0]
            res += len_g/len * en

        return res

    def entropy(self, X: pd.DataFrame, feature: Mapping, y_col: str, split: bool = True) ->float:
        """
        Returns entropy score of the give feature
        :param X: data
        :param feature: the feature you want to use to get compute entropy score
        :param y_col: name of the label column in X
        :return:
        """
        def split_entropy(y: pd.Series):
            ps = y.value_counts(normalize=True)
            return -(ps * np.log2(ps)).sum()

        len = X.shape[0]
        splits = self.split_on_feature(X, feature, y_col) if split else [X]

        res = 0
        for split in splits:
            en = split_entropy(split[y_col])
            len_g = split.shape[0]
            res += len_g/len * en

        return res



    def process_features(self, X: pd.DataFrame, y_col: str)->Sequence[Mapping]:
        """
        :param X: data
        :param y_col: name of the label column in X
        :return:
        """
        features = []
        for n,t in X.dtypes.items():
            if n == y_col:
                continue
            f = {'name': n, 'dtype': t}
            features.append(f)
        return features



