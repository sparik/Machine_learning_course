#!/usr/bin/env python3
"""
Homework 3 of Machine Learning Course

Implement polynomial_featurize(X, degree)
Implement TODO's

Example Usage:
    python3 homework2.py --user arsen

Make sure to test before submiting:
    python3 homework2.py test
"""
from __future__ import print_function  # If you want to run with python2
import argparse
import getpass
import matplotlib.pyplot as plt
import numpy as np
import os


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.set_defaults(function=main)
    parser.add_argument('--user', default=getpass.getuser(),
                        help='Override system username with something else to '
                             'be include in the output file.')
    subs = parser.add_subparsers()
    test_parser = subs.add_parser('test')
    test_parser.set_defaults(function=test_function_signatures)
    args = parser.parse_args(*argument_array)
    return args


def main(args):
    """Some different values of arguments to test the function on"""

    args1 = {"num_trials": 1000, "num_points": 40, "degree": 20, "beta_star": [10, 4, -0.01, -5. / 6, 0, 1. / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 0.01}

    args3 = {"num_trials": 1000, "num_points": 200, "degree": 20, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 0.01}

    args2 = {"num_trials": 1000, "num_points": 40, "degree": 20, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 10}

    args4 = {"num_trials": 1000, "num_points": 200, "degree": 20, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 10}

    args5 = {"num_trials": 1000, "num_points": 40, "degree": 1, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 0.01}

    args6 = {"num_trials": 1000, "num_points": 200, "degree": 1, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 0.01}

    args7 = {"num_trials": 1000, "num_points": 40, "degree": 5, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 0.01}

    args8 = {"num_trials": 1000, "num_points": 200, "degree": 5, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 0.01}

    args9 = {"num_trials": 1000, "num_points": 40, "degree": 5, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 1}

    args10 = {"num_trials": 1000, "num_points": 200, "degree": 5, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
              "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 1}

    # TODO add more arguments

    total, bias2, var = bias_variance(**args10)

    print("total= {}, bias^2= {}, var= {}".format(total, bias2, var))


def polynomial_featurize(X, degree):
    """
    param X: a one dimensional numpy array, i.g. [x1, x2, ..., xN]
    param degree: integer - degree of the polynomial wanted
    return: a 2D np_array where row i is 1x(degree+1) np_array representing the
        polynomial. e.g. [[1, x1, x1**2, ...], [1, x2, x2**2, ...], ...]
    """
    return np.column_stack(np.power(X, d) for d in range(degree + 1))


def decompose(beta_list, beta_star):
    """
    param beta_list: a list of 1D np_arrays - beta_hats
    param beta_star: a 1D np_array
        *NOTE that beta_hat and beta_star have different dimensions
        *you need to come up with a way of comparing them
    return: a 3D tuple (total_error, bias_squarred, variance)
    E[||beta_hat-beta_star||^2] = (E[beta_star - beta_hat ])^2 + E[beta_hat^2] - (E[beta_hat])^2
        *MAKE SURE you understand that each of the summations above is a number
        *UNDERSTAND that E[] is over the random variable beta_hat which depends on the random
            variable X - the data. Since we generate X many times, randomly, then beta_hat is
            a random varialbe. We are simulating its mean by trying many times.

    """

    d1 = len(beta_list[0])
    d2 = len(beta_star)

    vlen = max(d1, d2)

    if d2 < vlen:
        beta_star = np.pad(beta_star, (0, vlen - d2), 'constant')

    if d1 < vlen:
        beta_list = [np.pad(beta_hat, (0, vlen - d1), 'constant') for beta_hat in beta_list]

    diff_avg = np.sum((beta_star - beta_hat) for beta_hat in beta_list) / len(beta_list)
    bias_squarred = np.inner(diff_avg, diff_avg)


    beta_hat_avg = np.sum((beta_hat) for beta_hat in beta_list) / len(beta_list)
    E_beta_hat_sqr = np.inner(beta_hat_avg, beta_hat_avg)


    beta_hat_sqr_avg = np.sum((np.inner(beta_hat, beta_hat)) for beta_hat in beta_list) / len(beta_list)
    E_sqr_beta_hat = beta_hat_sqr_avg


    variance = E_sqr_beta_hat - E_beta_hat_sqr

    return variance + bias_squarred, bias_squarred, variance

## generates num_points (X, Y) pairs, X is generated uniformly in [x_left_lim, x_right_lim]
## Yi is generated from normal distribution with mean (beta_star*X[i])
## then beta_hat is calculated from (X, Y, l) and returned
def try_fit(num_points, degree, beta_star, sigma, x_left_lim=0, x_right_lim=100, l=1):
    X = np.random.uniform(x_left_lim, x_right_lim, num_points)
    poly_X = polynomial_featurize(X, degree) # (n, degree + 1)
    

    d = max(degree + 1, len(beta_star))

    temp_poly_X = polynomial_featurize(X, d - 1)

    if len(beta_star) < d:
        beta_star = np.pad(beta_star, (0, d - len(beta_star)), 'constant')

    Y = [np.random.normal(np.inner(beta_star, x), np.sqrt(sigma)) for x in temp_poly_X]
    Y = np.array(Y).reshape((len(Y), 1))


    beta_hat = fit_ridge_regression(poly_X, Y, l)
    return beta_hat

def bias_variance(num_trials, num_points, degree, beta_star, sigma, x_left_lim=0,
                  x_right_lim=100, l=1, num_red_lines=10):
    """
    param num_trials: integer - number of times we generate X given beta_star. We generate
        multiple X in order to understand the distribution of beta_hat.
    param num_points: integer - number of points in the data
    param degree: integer - degree of polynomial we use for fit_ridge_regression
    param beta_star: 1D np_array - beta_star. This is our REAL function with which poits are 
        generated. We are trying to approximate this function.
    param sigma: double - variane of normal error, when generating the point. The smaller the sigma
        is, the closer points are to the REAL function.
    param x_left_lim, x_right_lim: doubles - interval for generating x values
    param l: regularization lambda
    param num_red_lines: integer, number for red lines to show on the plot

    """
    # : you don't necessarily need the below, they are just here to make the code compile
    # and give you some idea
    beta_list = []
    X = np.random.uniform(x_left_lim, x_right_lim, num_points)

    poly_X = polynomial_featurize(X, len(beta_star) - 1)
    Y = poly_X.dot(beta_star)
    #  for num_trials times, generate X and Y given the params above. Compute beta_hat for
    # each of them and add that to beta_list

    beta_list = [try_fit(num_points, degree, beta_star, sigma, x_left_lim, x_right_lim, l) for trial in range(num_trials)]

    # README.
    # We provide the code below to help you visualize bias and variance, you don't need to
    #  change this.
    #
    plt.clf()
    line_x = np.arange(x_left_lim, x_right_lim, 0.1)
    beta_list2 = beta_list[::int(num_trials / num_red_lines)]
    for beta in beta_list2:
        line_y = [polynomial_featurize(x, degree).dot(beta).item(0, 0) for x in line_x]
        plt.plot(line_x, line_y, color='red')

    line_y = [polynomial_featurize(x, len(beta_star) - 1).dot(beta_star) for x in line_x]
    plt.plot(line_x, line_y, color='green')
    plt.scatter(X, Y)
    axes = plt.gca()
    axes.set_ylim([np.min(line_y), np.max(line_y)])
    plt.title('user: ' + getpass.getuser())
    plot_name = os.path.splitext(os.path.basename(__file__))[0] + '.png'
    plt.savefig(plot_name, dpi=320, bbox_inches='tight')
    return decompose(beta_list, beta_star)


def fit_ridge_regression(X, Y, l=1):
    """
    :param X: A matrix, where each row is a data element (X)
    :param Y: A list of responses for each of the rows (y)
    :param l: ridge variable
    :return: An np_array containing the hyperplane equation (beta)
    """
    # Implement to specification (you should have implemented similar function before)
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + l*np.identity(X.shape[1])), X.T), Y)


def test_function_signatures(args):
    total, bias, variance = decompose([np.array([1, 2]), np.array([2, 3])],
                                      np.array([1., 2.5]))
    assert np.abs(total - 0.75) < 1e-10
    assert np.abs(bias - 0.25) < 1e-10
    assert np.abs(variance - 0.50) < 1e-10

    poly_vec = polynomial_featurize(np.array([1, 2]), 3)
    assert poly_vec.shape == (2, 4)
    assert list(poly_vec[1, :]) == [1, 2, 4, 8]


if __name__ == '__main__':
    args = parse_args()
    args.function(args)