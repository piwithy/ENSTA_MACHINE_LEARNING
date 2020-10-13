import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X, y):
    """ computes the cost of using theta as the
    parameter for logistic regression."""

    # Initialize some useful values
    m, n = X.shape  # number of training examples and parameters
    theta = theta.reshape((n, 1))  # due to the use of fmin_tnc

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #
    h_theta = sigmoid(X @ theta)
    inner = -(y * np.log(h_theta) + (1 - y) * np.log(1-h_theta))

    # =============================================================

    return sum(inner) / m
