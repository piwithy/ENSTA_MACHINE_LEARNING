from sigmoid import sigmoid


def sigmoidGradient(z):
    """computes the gradient of the sigmoid function
    evaluated at z. This should work regardless if z is a matrix or a
    vector. In particular, if z is a vector or matrix, you should return
    the gradient for each element."""

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of the sigmoid function evaluated at
    #               each value of z (z can be a matrix, vector or scalar).

    sigma = sigmoid(z)

    g = sigma * (1 - sigma)
    # g = np.dot(sigmoid(z), (1 - sigmoid(z)).T)

    # =============================================================

    return g
