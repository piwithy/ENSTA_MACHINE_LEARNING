import time

def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear 
       regression to fit the data points in X and y
    """
    m = y.size
    J = 0.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    t_start = time.time()
    for i in range(m):
        h_theta = theta[0][0] + theta[1][0] * X[i][1]
        J += (h_theta - y[i]) ** 2
    J *= 1 / (2 * m)
    t_stop = time.time()
    # print("J calculation over " + str(m) + " items: ~" + str((t_stop-t_start)/1000) + "ms")
    #   ============================================================

    return J
