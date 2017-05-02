#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import logistic_regression
from logistic_regression import logistic_predict
from random_forest import RandomForest
from decision_tree import DecisionTree


def accuracy_score(Y_true, Y_predict):
    correct = 0
    for i in range(len(Y_true)):
        if Y_true[i] == Y_predict[i]:
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
    n, d = X.shape
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
        dtree_predictedY = dtree.predict(testX)
        dtree_accuracy = accuracy_score(testY, dtree_predictedY)
        dtree_accuracies.append(dtree_accuracy)

        # train random forest
        forest = RandomForest(10, 100)
        forest.fit(trainX, trainY)
        forest_predictedY = forest.predict(testX)[0]
        forest_accuracy = accuracy_score(testY, forest_predictedY)
        forest_accuracies.append(forest_accuracy)

        log_beta = logistic_regression(trainX, trainY, step_size=1e-1, max_steps=100)
        log_predictedY = logistic_predict(testX, log_beta)
        log_accuracy = accuracy_score(testY, log_predictedY)
        log_accuracies.append(log_accuracy)


    # compute the training accuracy of the model
    meanDecisionTreeAccuracy = np.mean(dtree_accuracies)
    stddevDecisionTreeAccuracy = np.std(dtree_accuracies)
    meanLogisticRegressionAccuracy = np.mean(log_accuracies)
    stddevLogisticRegressionAccuracy = np.std(log_accuracies)
    meanRandomForestAccuracy = np.mean(forest_accuracies)
    stddevRandomForestAccuracy = np.std(forest_accuracies)

    # make certain that the return value matches the API specification
    stats = np.zeros((3, 2))
    stats[0, 0] = meanDecisionTreeAccuracy
    stats[0, 1] = stddevDecisionTreeAccuracy
    stats[1, 0] = meanRandomForestAccuracy
    stats[1, 1] = stddevRandomForestAccuracy
    stats[2, 0] = meanLogisticRegressionAccuracy
    stats[2, 1] = stddevLogisticRegressionAccuracy
    return stats


# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluate_performance()
    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Random Forest Tree Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print("Logistic Reg. Accuracy = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.
