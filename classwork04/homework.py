from dataclasses import dataclass

import numpy as np

class PJ_Cart_Tree:
    def __init__(self, max_depth=4, acceptable_impurity=0.2):
        self.max_depth = max_depth
        self.acceptable_impurity=acceptable_impurity

    def split_node(self, x, y, feature, threshold):
        x_l = []
        y_l = []
        x_r = []
        y_r = []
        for feature_set, classification in zip(x, y):
            if feature_set[feature] > threshold:
                x_r.append(feature_set)
                y_r.append(classification)
            else:
                x_l.append(feature_set)
                y_l.append(classification)
        return np.asarray(x_l), np.asarray(y_l, dtype=np.int64), np.asarray(
                x_r), np.asarray(y_r, dtype=np.int64)

    def gini_impurity(self, y):
        instances = np.bincount(y)
        total = np.sum(instances)
        p = instances / total
        return 1.0 - np.sum(np.power(p, 2))

    def get_class(self, y):
        instances = np.bincount(y)
        return np.argmax(instances)

    def get_score_for_split(self, y, y_l, y_r, impurity_measure):
        left_score = impurity_measure(y_l) * y_l.shape[0] / y.shape[0]
        right_score = impurity_measure(y_r) * y_r.shape[0] / y.shape[0]
        return left_score + right_score

    def cart_split(self, x, y, granulation, impurity_measure):
        x_l_best = y_l_best = x_r_best = y_r_best = None
        score_best = feature_best = threshold_best = None
        for feature in range(x.shape[1]):
            start = np.min(x[:, feature])
            end = np.max(x[:, feature])
            step = (end - start) / granulation
            for threshold in np.arange(start, end, step):
                x_l, y_l, x_r, y_r = self.split_node(x, y, feature, threshold)
                score = self.get_score_for_split(y, y_l, y_r, impurity_measure)
                if score_best is None or score_best > score:
                    x_l_best = x_l
                    y_l_best = y_l
                    x_r_best = x_r
                    y_r_best = y_r
                    score_best = score
                    feature_best = feature
                    threshold_best = threshold
        return x_l_best, y_l_best, x_r_best, y_r_best, score_best, feature_best, threshold_best

    def grow_tree(self, x, y, max_depth, acceptable_impurity, granulation=10, curr_depth=0):
        node = dict(
                left = None,
                right = None,
                feature = None,
                threshold = None,
                gini = self.gini_impurity(y),
                node_class = self.get_class(y)
        )
        if curr_depth == max_depth or acceptable_impurity > node['gini']:
            return node
        x_l, y_l, x_r, y_r, score, feature, threshold = self.cart_split(x, y, granulation, self.gini_impurity)
        node['feature'] = feature
        node['threshold'] = threshold
        node['left'] = self.grow_tree(x_l, y_l, max_depth, acceptable_impurity, granulation, curr_depth=curr_depth+1)
        node['right'] = self.grow_tree(x_r, y_r, max_depth, acceptable_impurity, granulation, curr_depth=curr_depth+1)
        return node

    def fit(self, x, y):
        # print(self.get_class(y))
        self.tree = self.grow_tree(x, y, self.max_depth, self.acceptable_impurity)

    def descend_tree(self, x, node):
        if node['threshold'] is None:
            return node['node_class']
        if x[node['feature']] <= node['threshold']:
            return self.descend_tree(x,node['left'])
        else:
            return self.descend_tree(x, node['right'])

    def predict(self, x):
        return self.descend_tree(x, self.tree)
