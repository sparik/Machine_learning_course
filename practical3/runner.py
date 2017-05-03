#!/usr/bin/python3

"""
when run, this module compares performances of
decision tree, random forest, and logistic regression
"""

import numpy as np
from logistic_regression import logistic_regression
from logistic_regression import logistic_predict
from random_forest import RandomForest
from decision_tree import DecisionTree


def accuracy_score(Y_true, Y_predict):
    """
    calculates accuracy score of prediction, [0, 1]
    :param Y_true: true values
    :param Y_predicted: predicted values
    """
    correct = 0
    for i, item in enumerate(Y_predict):
        if item == Y_true[i]:
            correct += 1
    return correct / len(Y_true)


def evaluate_performance():
    '''
    Evaluate the performance of decision trees and logistic regression,
    average over 1,000 trials of 10-fold cross validation

    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of logistic regression
      stats[1,1] = std deviation of logistic regression accuracy

    ** Note that your implementation must follow this API**
    '''

    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')

    X = data[:, 1:]
    Y = np.array(data[:, 0])
    n = X.shape[0]
    folds = 10

    dtree_accuracies = []
    forest_accuracies = []
    log_accuracies = []

    for trial in range(10):
        np.random.seed(13)
        idx = np.arange(n)
        np.random.shuffle(idx)
        X = X[idx]
        Y = Y[idx]

        print("trial", trial + 1)

        trainsz = int((folds - 1) / (folds) * len(X))

        trainX = X[:trainsz]
        testX = X[trainsz:]
        trainY = Y[:trainsz]
        testY = Y[trainsz:]

        # train decision tree
        dtree = DecisionTree(100)
        dtree.fit(trainX, trainY)
        dtree_predicted = dtree.predict(testX)
        dtree_accuracy = accuracy_score(testY, dtree_predicted)
        dtree_accuracies.append(dtree_accuracy)

        # train random forest
        forest = RandomForest(10, 100)
        forest.fit(trainX, trainY)
        forest_predicted = forest.predict(testX)[0]
        forest_accuracy = accuracy_score(testY, forest_predicted)
        forest_accuracies.append(forest_accuracy)

        log_beta = logistic_regression(trainX, trainY, step_size=1e-1, max_steps=100)
        log_predicted = logistic_predict(testX, log_beta)
        log_accuracy = accuracy_score(testY, log_predicted)
        log_accuracies.append(log_accuracy)


    # compute the training accuracy of the model
    mean_decision_tree_accuracy = np.mean(dtree_accuracies)
    stddev_decision_tree_accuracy = np.std(dtree_accuracies)
    mean_log_regression_accuracy = np.mean(log_accuracies)
    stddev_log_regression_accuracy = np.std(log_accuracies)
    mean_random_forest_accuracy = np.mean(forest_accuracies)
    stddev_random_forest_accuracy = np.std(forest_accuracies)

    # make certain that the return value matches the API specification
    results = np.zeros((3, 2))
    results[0, 0] = mean_decision_tree_accuracy
    results[0, 1] = stddev_decision_tree_accuracy
    results[1, 0] = mean_random_forest_accuracy
    results[1, 1] = stddev_random_forest_accuracy
    results[2, 0] = mean_log_regression_accuracy
    results[2, 1] = stddev_log_regression_accuracy
    return results


# Do not modify from HERE...
if __name__ == "__main__":
    results = evaluate_performance()
    print("Decision Tree Accuracy = ", results[0, 0], " (", results[0, 1], ")")
    print("Random Forest Tree Accuracy = ", results[1, 0], " (", results[1, 1], ")")
    print("Logistic Reg. Accuracy = ", results[2, 0], " (", results[2, 1], ")")
# ...to HERE.
