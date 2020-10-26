import numpy as np

from sigmoid import sigmoid


def predictNeuralNetwork(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

    # Useful values
    m, _ = X.shape
    num_labels, _ = Theta2.shape
    p = np.zeros((m, 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a
    #               vector containing labels between 1 to num_labels.
    #
    # Hint: The max function might come in useful. In particular, the np.argmax
    #       function can return the index of the max element, for more
    #       information see 'numpy.argmax' on the numpy website. If your examples
    #       are in rows, then, you can use np.argmax(probs, axis=1) to obtain the
    #       max for each row.
    #

    # =========================================================================

    in_1 = np.ones((_, X.shape[0]))
    in_1[:-1, :] = np.dot(Theta1, X.T)

    z_2 = np.dot(Theta1, X.T)
    a_2 = np.ones((_, X.shape[0]))
    a_2[1:, :] = sigmoid(z_2)

    z_3 = np.dot(Theta2, a_2)
    a_3 = sigmoid(z_3)

    p = np.argmax(a_3, axis=0) + 1

    # =========================================================================

    return p
