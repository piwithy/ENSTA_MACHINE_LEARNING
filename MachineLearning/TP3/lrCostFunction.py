import numpy as np
from sigmoid import sigmoid


def lrCostFunction(theta, X, y, Lambda):
    """computes the cost of using
    theta as the parameter for regularized logistic regression.
    """

    # preambule
    m, n = X.shape  # 5,4
    theta = theta.reshape((n, 1))  # (4,1)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #
    # Hint: The computation of the cost function and gradients can be
    #       efficiently vectorized. For example, consider the computation
    #
    #           sigmoid(X @ theta) or np.dot(X, theta)
    #
    #       Each row of the resulting matrix will contain the value of the
    #       prediction for that example. You can make use of this to vectorize
    #       the cost function and gradient computations. 
    #
    term1 = -(y.T @ (np.log(sigmoid(X @ theta))))
    term2 = (1 - y).T @ (np.log(1 - sigmoid(X @ theta)))
    term3 = (np.linalg.norm(theta[1:]) ** 2) * (Lambda / (2 * m))
    J = (1 / m) * (term1 - term2) + term3

    # =============================================================

    return J
