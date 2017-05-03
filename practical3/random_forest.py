"""
this module contains the RandomForest class
"""

import numpy as np
from decision_tree import DecisionTree

class RandomForest(object):
    """
    RandomForest a class, that represents Random Forests.

    :param num_trees: Number of trees in the random forest
    :param max_tree_depth: maximum depth for each of the trees in the forest.
    :param ratio_per_tree: ratio of points to use to train each of
        the trees.
    """
    def __init__(self, num_trees, max_tree_depth, ratio_per_tree=0.5):
        self.num_trees = num_trees
        self.max_tree_depth = max_tree_depth
        self.ratio_per_tree = ratio_per_tree
        self.trees = []

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """

        if not isinstance(X, list):
            X = X.tolist()
        if not isinstance(Y, list):
            Y = Y.tolist()

        N = len(X)

        for _ in range(self.num_trees):
            sz = int(N*self.ratio_per_tree)
            idx = np.random.choice(N, sz, replace=True)
            trainX = [X[i] for i in idx]
            trainY = [Y[i] for i in idx]
            tree = DecisionTree(self.max_tree_depth)
            tree.fit(trainX, trainY)
            self.trees.append(tree)

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: (Y, conf), tuple with Y being 1 dimension python
        list with labels, and conf being 1 dimensional list with
        confidences for each of the labels.
        """

        Ys = [tree.predict(X) for tree in self.trees]

        Y = []
        conf = []
        for pred in range(len(Ys[0])):
            vals = [Ys[tree][pred] for tree in range(len(Ys))]
            vals = sorted(vals)

            mode = vals[0]
            maxcnt = 1
            curcnt = 1
            for i in range(1, len(vals)):
                if vals[i] != vals[i - 1]:
                    if curcnt > maxcnt:
                        mode = vals[i - 1]
                        maxcnt = curcnt
                    curcnt = 1
                else:
                    curcnt += 1

            if curcnt > maxcnt:
                mode = vals[-1]
                maxcnt = curcnt

            Y.append(mode)
            conf.append(maxcnt / self.num_trees)


        return (Y, conf)
