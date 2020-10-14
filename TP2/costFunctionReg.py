import numpy as np
from sigmoid import sigmoid


def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    m, n = X.shape  # number of training examples and parameters
    theta = theta.reshape((n, 1))  # due to the use of fmin_tnc

    J = 0.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #h_theta = sigmoid(X @ theta)
    #inner_left = sum(-(y * np.log(h_theta) + (1 - y) * np.log(1 - h_theta))) / m
    #inner_right = (Lambda / (2 * m)) * (np.linalg.norm(theta[:1]) ** 2)
    #J = inner_left + inner_right
    term1 = -(y.T @ (np.log(sigmoid(X@theta))))
    term2 = (1-y).T@(np.log(1-sigmoid(X@theta)))
    term3 = (np.linalg.norm(theta[1:])**2) * (Lambda/(2*m))
    J = (1/m) * (term1-term2) + term3
    # =============================================================

    # =============================================================

    return sum(J)
