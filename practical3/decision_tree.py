#!/usr/bin/python

from dtOmitted import DecisionNode
from dtOmitted import build_tree
import numpy as np

class DecisionTree(object):
    """
    DecisionTree class, that represents one Decision Tree

    :param max_tree_depth: maximum depth for this tree.
    """

    def __init__(self, max_tree_depth):
        self.max_depth = max_tree_depth

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        if not isinstance(X, list):
            X = X.tolist()
        if not isinstance(Y, list):
            Y = Y.tolist()
            
        data = [X[i][:] + [Y[i]] for i in range(len(X))]
        self.root = build_tree(data, max_depth=self.max_depth)
                

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: Y - 1 dimension python list with labels
        """
        if not isinstance(X, list):
            X = X.tolist()

        return [self.predict_one(self.root, row) for row in X]

    def predict_one(self, node, x):
        if node.is_leaf == False:
            xval = x[node.column]
            val = node.value
            if type(val) == int or type(val) == float:
                if xval > val:
                    return self.predict_one(node.true_branch, x)
                else:
                    return self.predict_one(node.false_branch, x)
            else:
                if xval == val:
                    return self.predict_one(node.true_branch, x)
                else:
                    return self.predict_one(node.false_branch, x)
        else:
            return node.result


def main():
    data = [['slashdot', 'USA', 'yes', 18, 'None'],
            ['google', 'France', 'yes', 23, 'Premium'],
            ['reddit', 'USA', 'yes', 24, 'Basic'],
            ['kiwitobes', 'France', 'yes', 23, 'Basic'],
            ['google', 'UK', 'no', 21, 'Premium'],
            ['(direct)', 'New Zealand', 'no', 12, 'None'],
            ['(direct)', 'UK', 'no', 21, 'Basic'],
            ['google', 'USA', 'no', 24, 'Premium'],
            ['slashdot', 'France', 'yes', 19, 'None'],
            ['reddit', 'USA', 'no', 18, 'None'],
            ['google', 'UK', 'no', 18, 'None'],
            ['kiwitobes', 'UK', 'no', 19, 'None'],
            ['reddit', 'New Zealand', 'yes', 12, 'Basic'],
            ['slashdot', 'UK', 'no', 21, 'None'],
            ['google', 'UK', 'yes', 18, 'Basic'],
            ['kiwitobes', 'France', 'yes', 19, 'Basic']]

    X = [row[0:-1] for row in data]
    Y = [row[-1] for row in data]

    tree = DecisionTree(100)
    tree.fit(X, Y)
    predictedY = tree.predict(X)

    for i in range(len(Y)):
        assert Y[i] == predictedY[i]
    #tree = build_tree(data)
    #print_tree(tree)

if __name__ == '__main__':
    main()
