#!/usr/bin/python3

"""
this module contains the DecisionNode class
and a function for building a decision tree
"""


class DecisionNode(object):
    """
    README
    DecisionNode is a building block for Decision Trees.
    DecisionNode is a python class representing a  node in our decision tree
    node = DecisionNode()  is a simple usecase for the class
    you can also initialize the class like this:
    node = DecisionNode(column = 3, value = "Car")
    In python, when you initialize a class like this, its __init__ method is called
    with the given arguments. __init__() creates a new object of the class type, and initializes its
    instance attributes/variables.
    In python the first argument of any method in a class is 'self'
    Self points to the object which it is called from and corresponds to 'this' from Java

    """

    def __init__(self,
                 column=None,
                 value=None,
                 false_branch=None,
                 true_branch=None,
                 current_results=None,
                 is_leaf=False,
                 result=None):
        self.column = column
        self.value = value
        self.false_branch = false_branch
        self.true_branch = true_branch
        self.current_results = current_results
        self.is_leaf = is_leaf
        self.result = result
        max_cnt = -1
        for val in current_results.keys():
            cnt = current_results[val]
            if cnt > max_cnt:
                max_cnt = cnt
                self.result = val


def dict_of_values(data):
    """
    param data: a 2D Python list representing the data. Last column of data is Y.
    return: returns a python dictionary showing how many times each value appears in Y

    for example
    data = [[1,'yes'],[1,'no'],[1,'yes'],[1,'yes']]
    dict_of_values(data)
    should return {'yes' : 3, 'no' :1}
    """
    results = {}
    for row in data:
        res = row[len(row) - 1]
        if res in results:
            results[res] += 1
        else:
            results[res] = 1
    return results


def divide_data(data, feature, value):
    """
    this function dakes the data and divides it in two parts by a line. A line
    is defined by the feature we are considering (feature_column) and the target
    value. The function returns a tuple (data1, data2) which are the desired parts of the data.
    For int or float types of the value, data1 have all the data with values >= feature_val
    in the corresponding column and data2 should have rest.
    For string types, data1 should have all data with values == feature val and data2 should
    have the rest.

    param data: a 2D Python list representing the data. Last column of data is Y.
    param feature_column: an integer index of the feature/column.
    param feature_val: can be int, float, or string
    return: a tuple of two 2D python lists
    """
    if isinstance(value, (int, float)):
        true_data = [row for row in data if row[feature] > value]
        false_data = [row for row in data if row[feature] <= value]
    else:
        true_data = [row for row in data if row[feature] == value]
        false_data = [row for row in data if row[feature] != value]
    return false_data, true_data

def gini_impurity(data):

    """
    Given two 2D lists of compute their gini_impurity index.
    Remember that last column of the data lists is the Y
    Lets assume y1 is y of data1 and y2 is y of data2.
    gini_impurity shows how diverse the values in y1 and y2 are.
    gini impurity is given by

    N1*sum(p_k1 * (1-p_k1)) + N2*sum(p_k2 * (1-p_k2))

    where N1 is number of points in data1
    p_k1 is fraction of points that have y value of k in data1
    same for N2 and p_k2


    param data1: A 2D python list
    param data2: A 2D python list
    return: a number - gini_impurity
    """

    if not data:
        return 0

    value_counts = dict_of_values(data)

    gini = 0.0
    for value in value_counts:
        percentage_of_value = value_counts[value] / len(data)
        gini += percentage_of_value * (1.0 - percentage_of_value)

    gini *= len(data)

    return gini


def build_tree(data, current_depth=0, max_depth=1e10):
    """
    build_tree is a recursive function.
    What it does in the general case is:
    1: find the best feature and value of the feature to divide the data into
    two parts
    2: divide data into two parts with best feature, say data1 and data2
        recursively call build_tree on data1 and data2. this should give as two
        trees say t1 and t2. Then the resulting tree should be
        DecisionNode(...... true_branch=t1, false_branch=t2)


    In case all the points in the data have same Y we should not split any more,
    and return that node
    For this function we will give you some of the code so its not too hard for you ;)

    param data: param data: A 2D python list
    param current_depth: an integer. This is used if we want to limit the numbr of layers in the
        tree
    param max_depth: an integer - the maximal depth of the representing
    return: an object of class DecisionNode

    """
    if not data:
        return DecisionNode(is_leaf=True)

    if current_depth == max_depth:
        return DecisionNode(current_results=dict_of_values(data), is_leaf=True)

    if len(dict_of_values(data)) == 1:
        return DecisionNode(current_results=dict_of_values(data), is_leaf=True)

    #Below are the attributes of the best division that you need to find.
    #You need to update these when you find a division which is better
    best_gini = 1e10
    best_column = None
    best_value = None
    #best_split is tuple (data1,data2) which shows the two datas for the best divison so far
    best_split = None

    #You need to find the best feature to divide the data
    #For each feature and each possible value of the feature compute the
    # gini number for that division. You need to find the feature that minimizes
    # gini number. Remember that last column of data is Y
    # Think how you can use the divide_data and gini_impurity functions you wrote
    # above

    for feature in range(len(data[0]) - 1):
        unique_values = [data[i][feature] for i in range(len(data))]
        unique_values = set(unique_values)
        unique_values = list(unique_values)
        # I think it's already sorted but just in case
        unique_values = sorted(unique_values)

        for value in unique_values:
            (false_data, true_data) = divide_data(data, feature, value)
            if not true_data or not false_data:
                continue
            cur_impurity = gini_impurity(false_data) + gini_impurity(true_data)
            if cur_impurity < best_gini:
                best_gini = cur_impurity
                best_column = feature
                best_value = value
                best_split = (false_data, true_data)

    #recursively call build tree, construct the correct return argument and return
    false_tree = build_tree(best_split[0], current_depth + 1, max_depth)
    true_tree = build_tree(best_split[1], current_depth + 1, max_depth)

    return DecisionNode(column=best_column,
                        value=best_value,
                        false_branch=false_tree,
                        true_branch=true_tree,
                        current_results=dict_of_values(data))



def print_tree(tree, indent=''):
    """
    :param tree: the root of decision tree, an instance of DecisionNode class
    :param indent: string to print before the tree, for indentation
    """

    # Is this a leaf node?
    if tree.is_leaf:
        print(str(tree.current_results))
    else:
        # Print the criteria
        #         print (indent+'Current Results: ' + str(tree.current_results))
        print('Column ' + str(tree.column) + ' : ' + str(tree.value) + '? ')

        # Print the branches
        print(indent + 'True->', end="")
        print_tree(tree.true_branch, indent + '  ')
        print(indent + 'False->', end="")
        print_tree(tree.false_branch, indent + '  ')


def main():
    """
    test implementation
    """
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

    tree = build_tree(data)
    print_tree(tree)


if __name__ == '__main__':
    main()
