"""
this module contains implementation of logistic regression
"""

import numpy as np

def normalize_data(X, l):
    Xs = np.copy(X)

    l = [l]*Xs.shape[1]
    l[0] = 0
    fstd = [0.0]*Xs.shape[1]
    fmean = [0.0]*Xs.shape[1]

    for i in range(1, Xs.shape[1]):
        fstd[i] = np.std(Xs[:, i])
        fmean[i] = np.mean(Xs[:, i])
        if fstd[i] < 0.001:
            fstd[i] = 1
        Xs[:, i] = (Xs[:, i] - fmean[i]) / fstd[i]
        l[i] = l[i] / (fstd[i] * fstd[i])

    return Xs, l, fstd, fmean


def rescale_beta(beta, fmean, fstd):
    beta[0] = beta[0] - sum(fmean[i] * beta[i] / fstd[i] for i in range(1, len(fmean)))

    for i in range(1, len(fmean)):
        beta[i] = beta[i] / fstd[i]


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def normalized_gradient(X, Y, beta, l):
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param beta: value of beta (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: normalized gradient, i.e. gradient normalized according to data
    """
    m = len(X)
    n = len(beta)
    beta = beta.reshape(n, 1)
    grad = np.empty([n], dtype=float)

    for i in range(n):
        grad[i] = 2 * l[i] * beta[i]
        for j in range(m):
            grad[i] -= X[j, i] * Y[j] * (1 - sigmoid(Y[j] * np.dot(beta.T, X[j, :])))

    return grad / len(X)


def logistic_regression(X, Y, epsilon=1e-4, l=1, step_size=1e-3, max_steps=500):
    """
    Implement gradient descent using full value of the gradient.
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :param epsilon: approximation strength
    :param max_steps: maximum number of iterations before algorithm will
        terminate.
    :return: value of beta (1 dimensional np.array)
    """

    X1 = np.column_stack((np.ones(len(X)), X))

    (Xs, l, fstd, fmean) = normalize_data(X1, l)

    beta = np.zeros(Xs.shape[1])
    for _ in range(max_steps):
        gradient = normalized_gradient(Xs, Y, beta, l)
        if np.inner(gradient, gradient) < epsilon*epsilon:
            break
        beta = beta - step_size*gradient


    rescale_beta(beta, fmean, fstd)

    return beta

def logistic_predict(X, beta):
    X1 = np.column_stack(((np.ones(len(X))), X))
    probs = [sigmoid(np.dot(x, beta)) for x in X1]
    Y = [1 if prob > 0.5 else 0 for prob in probs]
    return Y

