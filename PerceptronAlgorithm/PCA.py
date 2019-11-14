# Predict:  y^i=sign(w˙⊤x˙i)
# Update rule:  w˙=w˙+ηx˙iyi  for all missclassified

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


def sign(x):
    # x is a scalar (number)
    # TODO: implement sign function. Sign function
    #       returns 1  if x is positive,
    #       returns -1 otherwise
    pass


def generate_X_dot(X):
    # X is a matrix size n*f, where:
    #   n is the number of sample x in X
    #   f is the number of features of a single sample x
    # TODO: Append a column of number 1 to the left of X
    pass


def predict_single_data(x_dot, w_dot):
    # x_dot is an array contains f+1 element, where
    #       f is the number of feature of x
    # w_dot is a column vector that has the same size with x_dot
    # TODO: implement function to predict label of x_dot
    pass


def predict(X_dot, w_dot):
    # X_dot is a matrix size n*(f+1), where:
    #       n is the number of sample x_dot in X_dot
    #       f is the number of feature of x in X
    # TODO: calculate y from data X_dot and weight w_dot
    pass


def update_w_dot(x_dot, w_dot, y, learning_rate):
    # x_dot is an array contains f+1 element, where
    #       f is the number of feature of x
    # w_dot is a column vector that has the same size with x_dot
    # y     is the corresponding label of current x_dot
    # learning_rate is a float
    # TODO: implement function to update w_dot
    pass


def train(X, y, epochs, learning_rate):
    # X      is a matrix size n*f, where:
    #        n is total sample x in X
    #        f is the number of feature of x in X
    # y      is an array of known labels corresponding
    #        to each sample x in X
    # epochs is total training loops over data set
    # learning_rate is eta in formal equation (lecture note)
    # TODO: generate X_dot from X

    w_dot = np.zeros(len(X[0])+1)

    for epoch in range(epochs):
        # TODO:
        # - predict label of every point x_dot in X_dot
        # - if this point is missclassified, update w_dot
        pass

    return w_dot


def train_verbose(X, y, epochs, learning_rate):

    # TODO: generate X_dot from X

    w_dot = np.zeros(len(X[0])+1)
    error_history = []

    for epoch in range(epochs):
        # TODO:
        # - predict label of every point x_dot_i
        # - if this point is missclassified, update w_dot
        # - remember to append total missclassified points
        #   to error_history in every epoch
        pass

    return w_dot, error_history


############################# Generate data set #############################
N = None
means = [[2, 2], [4, 4]]
cov = [[1, 0], [0, 1]]

N = 500
C = 2

X_0 = np.random.multivariate_normal(means[0], cov, N)
X_1 = np.random.multivariate_normal(means[1], cov, N)

X = np.concatenate((X_0, X_1), axis=0)
y = np.array([[-1]*N, [1]*N]).reshape(C*N,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

## ##
w_dot = train(X_train, y_train, learning_rate=0.1, max_epochs=20)
# TODO: calculate y_pred based on learned w_dot

print("Accuracy: %.2f %%" % (100*accuracy_score(y_test, y_pred)))


############################# Visulize #############################
    # TODO: Visulize X_test on 2D graph, colored by y_test
    # TODO: Visualize X_test on 2D graph, colored by y_pred
    # TODO: Visualize decision boundary (optional)

##########################
w_dot, error_history = train_verbose(X_train, y_train, max_epochs = 20, learning_rate=0.1)
# TODO: visualize error_history by the graph