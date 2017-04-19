#!/usr/bin/env python3
"""
Run regression on apartment data.
"""
from __future__ import print_function
import argparse
import pandas as pd
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import getpass


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.set_defaults(function=main)
    parser.add_argument('--user', default=getpass.getuser(),
                        help='Override system username with something else to '
                             'be include in the output file.')
    subs = parser.add_subparsers()
    test_parser = subs.add_parser('test')
    test_parser.set_defaults(function=test_function_signatures)
    parser.add_argument('--csv', default='yerevan_april_9.csv.gz',
                        help='CSV file with the apartment data.')
    args = parser.parse_args(*argument_array)
    return args


def featurize(apartment):
    """
    :param apartment: Apartment DataFrame row (a dictionary like object)
    :return: (x, y) tuple, where x is a numpy vector, and y is a number
    """

    features = [apartment['area'], apartment['num_bathrooms'], apartment['floor'], apartment['ceiling_height'], apartment['max_floor'], apartment['num_rooms']];

    for d in range(2):
        features_d = [pow(f, d + 2) for f in features]
        features.extend(features_d)

    

    house_condition = [0, 0, 0]

    if apartment['condition'] == 'good':
        house_condition[0] = 1
    if apartment['condition'] == 'newly repaired':
        house_condition[1] = 1
    if apartment['condition'] == 'zero condition':
        house_condition[2] = 1

    district = [0, 0, 0, 0, 0, 0, 0, 0]

    if apartment['district'] == 'Center':
        district[0] = 1
    elif apartment['district'] == 'Arabkir':
        district[1] = 1
    elif apartment['district'] == 'Avan':
        district[2] = 1
    elif apartment['district'] == 'Achapnyak':
        district[3] = 1
    elif apartment['district'] == 'Vahagni district':
        district[4] = 1
    elif apartment['district'] == 'Nor Norq':
        district[5] = 1
    elif apartment['district'] == 'Malatia-Sebastia':
        district[6] = 1
    else:
        district[7] = 1

    btype = [0, 0, 0, 0]

    if apartment['building_type'] == 'panel':
        btype[0] = 1
    if apartment['building_type'] == 'monolit':
        btype[1] = 1
    if apartment['building_type'] == 'stone':
        btype[2] = 1
    if apartment['building_type'] == 'other':
        btype[3] = 1

    streets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    if "Baghramyan" in apartment['street']:
        streets[0] = 1
    elif "Teryan" in apartment['street']:
        streets[1] = 1
    elif "Tumanyan" in apartment['street']:
        streets[2] = 1
    elif "Aram" in apartment['street']:
        streets[3] = 1
    elif "Mashtots" in apartment['street']:
        streets[4] = 1
    elif "Northern" in apartment['street']:
        streets[5] = 1
    elif "Komitas" in apartment['street']:
        streets[6] = 1
    elif "Amiryan" in apartment['street']:
        streets[7] = 1
    elif "Khorenatsi" in apartment['street']:
        streets[8] = 1
    elif "Sayat Nova" in apartment['street']:
        streets[9] = 1
    elif "Pushkin" in apartment['street']:
        streets[10] = 1
    else:
        streets[11] = 1

    features.extend(house_condition)
    features.extend(district)
    features.extend(btype)
    features.extend(streets)

    
    return features


def poly_featurize(apartment, degree=2):
    """
    :param apartment: Apartment DataFrame row (a dictionary like object)
    :return: (x, y) tuple, where x is a numpy vector, and y is a number
    """

    return [0]


def fit_ridge_regression(X, Y, l=0.1):
    """
    :param X: A numpy matrix, where each row is a data element (X)
    :param Y: A numpy vector of responses for each of the rows (y)
    :param l: ridge variable
    :return: A vector containing the hyperplane equation (beta)
    """
    li = l*np.identity(X.shape[1])
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + li), X.T), Y)



def cross_validate(X, Y, fitter, folds=5):
    """
    :param X: A numpy matrix, where each row is a data element (X)
    :param Y: A numpy vector of responses for each of the rows (y)
    :param fitter: A function that takes X, Y as parameters and returns beta
    :param folds: number of cross validation folds (parts)
    :return: list of corss-validation scores
    """
    scores = []
    # TODO: Divide X, Y into `folds` parts (e.g. 5)
    assert len(X) == len(Y)
    l = len(X)
    parts_indices = np.random.choice(l, (5, (int)(l / 5)), replace=False)

    X_parts = np.array([X[index] for index in parts_indices])
    Y_parts = np.array([Y[index] for index in parts_indices])

    for i in range(folds):
        # TODO: train on the rest
        # TODO: Add corresponding score to scores
        train_X = np.empty((0, X.shape[1]))
        train_Y = np.empty((0,))
        for j in range(folds):
            if j == i:
                continue
            train_X = np.concatenate((train_X, X_parts[j]))
            train_Y = np.concatenate((train_Y, Y_parts[j]))
        test_X = X_parts[i]
        test_Y = Y_parts[i]
        beta = fitter(train_X, train_Y)

        cur_score = np.sum([np.inner((np.inner(beta, test_X[ind]) - test_Y[ind]), (np.inner(beta, test_X[ind]) - test_Y[ind])) for ind in range(len(test_X))])

        scores.append(cur_score)


        
    return scores


def my_featurize(apartment):
    """
    This is the function we will use for scoring your implmentation.
    :param apartment: apartment row
    :return: (x, y) pair where x is feature vector, y is the response variable.
    """

    return np.array(featurize(apartment)), apartment['price']

def my_beta():
    """
    :return: beta_hat that you estimate.
    """
    return np.array([  4.92310603e+03,   8.60699111e+03,   3.36735855e+03,  -1.38977784e+04,
   3.88290359e+03,  -1.51152733e+04,  -6.92791742e+01,   5.29647386e+03,
  -5.80625521e+02,  -5.10789349e+04,  -3.01871057e+02,   9.33002086e+03,
   3.85459094e-01,  -2.30690921e+03,   2.42914931e+01,   1.80876331e+04,
   5.55487488e+00,  -1.59922588e+03,  -2.21924243e-08,   1.88510573e+01,
  -1.48035903e-04,  -9.78514167e+01,   3.58527653e-04,   1.06146615e+00,
   5.59127114e+03,   1.57663433e+04,  -1.13760436e+04,   3.15026323e+04,
   9.77319070e+03,  -9.77936676e+03,  -9.36358906e+03,   7.24724138e+03,
  -7.78171433e+03,  -8.51859263e+03,  -3.09823141e+03,  -6.81798971e+03,
   1.14857655e+04,   9.65126097e+02,   4.34866834e+03,  -1.68553965e+04,
   5.95249090e+03,  -4.74391887e+03,   2.25810537e+04,  -9.61511011e+03,
   1.17181910e+05,  -2.42624941e+04,  -3.97949960e+03,  -4.38237851e+04,
  -2.80446112e+03,  -1.03611351e+03,  -2.86131055e+04])


def main(args):

    df = pd.read_csv(args.csv)

    Y = np.array([df.iloc[line]['price'] for line in range(len(df))])

    X = np.array([featurize(df.iloc[x]) for x in range(len(df))])

    # TODO: Convert `df` into features (X) and responses (Y) using featurize

    print(fit_ridge_regression(X, Y))
    # TODO you should probably create another function to pass to `cross_validate`
    scores = cross_validate(X, Y, fit_ridge_regression)

    print(np.mean(scores))

    ans = np.sqrt(5 * np.mean(scores)/(len(X)))
    print(ans)


def test_function_signatures(args):
    apt = pd.Series({'price': 65000.0, 'condition': 'good', 'district': 'Center', 'max_floor': 9, 'street': 'Vardanants St', 'num_rooms': 3, 'region': 'Yerevan', 'area': 80.0, 'url': 'http://www.myrealty.am/en/item/24032/3-senyakanoc-bnakaran-vacharq-Yerevan-Center', 'num_bathrooms': 1, 'building_type': 'panel', 'floor': 4, 'ceiling_height': 2.7999999999999998})  # noqa
    x, y = my_featurize(apt)
    beta = my_beta()

    assert type(y) == float
    assert len(x.shape) == 1
    assert x.shape == beta.shape

if __name__ == '__main__':
    args = parse_args()
    args.function(args)
