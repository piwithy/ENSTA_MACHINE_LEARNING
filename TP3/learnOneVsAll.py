import numpy as np

from lrCostFunction import lrCostFunction
from lrCostGradient import lrCostGradient
from scipy.optimize import fmin_tnc


def learnOneVsAll(X, y, num_labels, Lambda):
    """trains multiple logistic regression classifiers and returns all
        the classifiers in a matrix all_theta, where the i-th row of all_theta
        corresponds to the classifier for label i
    """

    # Some useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n))

    # Set Initial theta
    initial_theta = np.zeros((n, 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter lambda.
    #
    #
    # Hint: You can use (y == c)*1 to obtain a vector of 1's and 0's that tell use
    #       whether the ground truth is true/false for this class.
    #
    # Note: For this assignment, we recommend using fmin_tnc to optimize the cost
    #       function. It is okay to use a for-loop (for c = 1:num_labels) to
    #       loop over the different classes.

    # première solution
    for i in range(1, num_labels + 1):
        print('Optimizing for handwritten number %d...' % i)
        y_1vsAll = (y == i) * 1

        # on garantit que y_1vsAll soit à deux dim: évite pas de prb avec la fonction fmin_tnc
        # y_1vsAll = np.atleast_2d(y_1vsAll).T

        result = fmin_tnc(lrCostFunction, fprime=lrCostGradient, x0=initial_theta, args=(X, y_1vsAll, Lambda),
                          disp=False)

        all_theta[i - 1, :] = result[0]

    print('Done!')

    # =========================================================================
    # This function will return all_theta 

    return all_theta
