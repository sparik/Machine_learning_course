import numpy as np

def normalize_data(X, l):
    Xs = np.copy(X)

    l = [l]*Xs.shape[1]
    l[0] = 0
    fstd = [0]*Xs.shape[1]
    fmean = [0]*Xs.shape[1]

    for i in range(1, Xs.shape[1]):
        fstd[i] = np.std(Xs[:, i])
        fmean[i] = np.mean(Xs[:, i])
        Xs[:, i] = (Xs[:, i] - fmean[i]) / fstd[i]
        l[i] = l[i] / (fstd[i] * fstd[i])

    return Xs, l, fstd, fmean

def rescale_beta(beta, fmean, fstd):
    beta[0] = beta[0] - sum(fmean[i] * beta[i] / fstd[i] for i in range(1, len(fmean)))

    for i in range(1, len(fmean)):
        beta[i] = beta[i] / fstd[i]

def fit_ridge_regression(X, Y, l):
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: value of beta (1 dimensional np.array)
    """
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + l*np.identity(X.shape[1])), X.T), Y)


def gradient_descent(X, Y, epsilon=1e-6, l=1, step_size=1e-4, max_steps=1000):
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

    (Xs, l, fstd, fmean) = normalize_data(X, l)

    beta = np.zeros(Xs.shape[1])
    for s in range(max_steps):
        gradient = normalized_gradient(Xs, Y, beta, l)
        if np.inner(gradient, gradient) < epsilon*epsilon:
            break
        beta = beta - step_size * gradient


    rescale_beta(beta, fmean, fstd)

    return beta


def normalized_gradient(X, Y, beta, l):
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param beta: value of beta (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: normalized gradient, i.e. gradient normalized according to data
    """
    beta = np.array(beta)

    return 1 / len(Y) * (l*beta - np.dot(X.T, Y-np.dot(X, beta)))


def stochastic_gradient_descent(X, Y, epsilon=0.0001, l=1, step_size=0.01,
                                max_steps=1000):
    """
    Implement gradient descent using stochastic approximation of the gradient.
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :param epsilon: approximation strength
    :param max_steps: maximum number of iterations before algorithm will
        terminate.
    :return: value of beta (1 dimensional np.array)
    """

    (Xs, l, fstd, fmean) = normalize_data(X, l)

    index = 0
    beta = np.ones(Xs.shape[1])
    for s in range(max_steps):
        gradient = normalized_gradient(Xs[index].reshape(1, Xs.shape[1]), Y[index:index + 1], beta, l)
        beta = beta - step_size * gradient
        if np.inner(gradient, gradient) < epsilon*epsilon:
            break

        index = index + 1
        if index >= len(Xs):
            index = 0

    rescale_beta(beta, fmean, fstd)

    return beta
