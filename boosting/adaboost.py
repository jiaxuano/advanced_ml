import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))

def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    ### BEGIN SOLUTION
    X_list = []
    y_list = []
    with open(filename, 'r') as f:
        for line in f:
            line_list = line.split(',')
            x = [float(i) for i in line_list[:-1]]
            y = int(line_list[-1])
            X_list.append(x)
            y_list.append(y)
    Y = np.array([-1 if i == 0 else i for i in y_list])
    X = np.array(X_list)
    return X, Y


def adaboost(X, y, num_iter, max_depth=1):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    """
    trees = []
    trees_weights = [] 
    N, _ = X.shape
    d = np.ones(N) / N

    ### BEGIN SOLUTION
    for i in range(num_iter):
        tree = DecisionTreeClassifier(max_depth=max_depth)
        tree.fit(X, y)
        trees.append(tree)
        pred = tree.predict(X)
        err = np.sum(d * (y != pred)) / np.sum(d)
        alpha = np.log((1-err)/err)
        trees_weights.append(alpha)
        d = d* np.exp(alpha * (y != pred))
    ### END SOLUTION
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    """
    # X input, y output
    N, _ =  X.shape
    y_pred = np.zeros(N)
    ## BEGIN SOLUTION
    for i in range(len(trees)):
        y_pred += trees[i].predict(X) * trees_weights[i]
    y = np.sign(y_pred)
    ### END SOLUTION
    return y
