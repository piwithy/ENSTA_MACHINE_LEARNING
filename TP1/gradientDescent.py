import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta, cost_history, theta_history = gradientDescent(X, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """
    # Initialize some useful values
    m = y.size  # number of training examples
    n = theta.size  # number of parameters
    cost_history = np.zeros(num_iters)  # cost over iters
    theta_history = np.zeros((n, num_iters))  # theta over iters
    for i in range(num_iters):
        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        # TODO imple -> formule 4.2.1
        summ = 0.
        for j in range(m):
            h_theta = theta[0][0] + theta[1][0] * X[j][1]
            summ += (h_theta - y[j])*X[j, :].reshape(2,1)
        theta -= (alpha / m) * summ

        cost_history[i] = computeCost(X, y, theta)
        theta_history[:, i] = theta.reshape((2,))

    #   ============================================================

    return theta, cost_history, theta_history
