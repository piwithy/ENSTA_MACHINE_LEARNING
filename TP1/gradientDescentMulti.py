import numpy as np
from computeCostMulti import computeCostMulti


def gradientDescentMulti(X, y, theta, alpha, num_iters):  
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """
    # Initialize some useful values
    m = y.size  # number of training examples
    n = theta.size # number of parameters
    cost_history = np.zeros(num_iters)
    theta_history = np.zeros((n,num_iters))


    for i in range(num_iters):
    #   ====================== YOUR CODE HERE ======================
    # Instructions: Perform a single gradient step on the parameter vector
    #               theta.
    #
    # Hint: While debugging, it can be useful to print out the values
    #       of the cost function (computeCost) and gradient here.
    #






        cost_history[i] = computeCostMulti(X, y, theta)
        theta_history[:,i] = theta.reshape((n,))
      
    #   ============================================================
        
    return theta, cost_history, theta_history
