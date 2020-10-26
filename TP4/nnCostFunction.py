import numpy as np

from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    """computes the cost and gradient of the neural network. The
  parameters for the neural network are "unrolled" into the vector
  nn_params and need to be converted back into the weight matrices.

  The returned parameter grad should be a "unrolled" vector of the
  partial derivatives of the neural network.
    """

    # Reshape nn_params back into the parameters theta1 and theta2, the weight matrices
    # for our 2 layer neural network
    # Obtain theta1 and theta2 back from nn_params

    theta1 = nn_params[0:(hidden_layer_size * (input_layer_size + 1))].reshape((input_layer_size + 1),
                                                                               hidden_layer_size).T
    theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):].reshape((hidden_layer_size + 1), num_labels).T

    # Setup some useful variables
    m, _ = X.shape

    # You need to return the following variables correctly
    J = 0
    theta1_grad = np.zeros(theta1.shape);
    theta2_grad = np.zeros(theta2.shape);

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.py
    #
    #	NOTE DE GLC: JE VOUS AI MIS LA CORRECTION DES PART 2 ET 3 dans la suite du code
    #   
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         theta1_grad and theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to theta1 and theta2 in theta1_grad and
    #         theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1...K. You need to map this vector into a 
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the 
    #               first time.
    #
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to theta1_grad
    #               and theta2_grad from Part 2.
    #
    # =========================================================================

    # Add ones to the X data matrix

    # Construct a 10xm "y" matrix with all zeros and only one "1" entry
    # note here if the hand-written digit is "0", then that corresponds
    # to a y- vector with 1 in the 10th spot (different from what the
    # homework suggests)
    y_matrix = np.zeros((num_labels, m))  # A compléter
    for i in range(m):
        y_matrix[y[i]-1, i] = 1
    a1 = np.ones((X.shape[0], X.shape[1] + 1))
    a1[:, 1:] = X
    z2 = np.dot(theta1, a1.T)
    a2 = np.ones((z2.shape[0] + 1, a1.shape[0]))
    a2[1:, :] = sigmoid(z2)

    z3 = np.dot(theta2, a2)
    a3 = sigmoid(z3).T
    inner1 = (y_matrix.T * np.log(a3))
    inner2 = (1 - y_matrix.T) * np.log(1 - a3)
    J = (1 / m) * np.sum(-inner1 - inner2)

    # Compute Cost

    # =========================================================================

    # Cost regularisation
    reg = (Lambda / (2 * m)) * (np.sum(np.square(theta1[:, 1:])) + np.sum(np.square(theta2[:, 1:])))
    J = J + reg

    # Gradients
    d3 = a3 - y_matrix.T  # 10x5000
    d2 = theta2[:, 1:].T.dot(d3.T) * sigmoidGradient(z2)  # 25x10 *10x5000 * 25x5000 = 25x5000

    delta1 = d2.dot(a1)  # 25x5000 * 5000x401 = 25x401
    delta2 = d3.T.dot(a2.T)  # 10x5000 *5000x26 = 10x26

    # Gradient regularisation
    theta1_grad = delta1 / m
    reg = (theta1[:, 1:] * Lambda) / m
    theta1_grad[:, 1:] = theta1_grad[:, 1:] + reg

    theta2_grad = delta2 / m
    reg = (theta2[:, 1:] * Lambda) / m
    theta2_grad[:, 1:] = theta2_grad[:, 1:] + reg

    # Unroll gradient
    grad = np.hstack((theta1_grad.T.ravel(), theta2_grad.T.ravel()))

    return J, grad
