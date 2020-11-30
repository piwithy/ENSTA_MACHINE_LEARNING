import numpy as np


def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear 
       regression to fit the data points in X and y
    """
    m = y.size
    J = 0.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    for i in range(m):
        h_theta = np.dot(X[i, :], theta)
        J += (h_theta - y[i]) ** 2
    J *= 1 / (2 * m)
    #   ============================================================

    return J
