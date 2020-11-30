import numpy as np


def featureNormalize(X):
    """
       returns a normalized version of X where
       the mean value of each feature is 0 and the standard deviation
       is 1. This is often a good preprocessing step to do when
       working with learning algorithms.
    """
    X_norm = np.zeros(X.shape)
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    # X_norm, mu, sigma = [],[],[]
    # ====================== YOUR CODE HERE ======================
    # Instructions: First, for each feature dimension, compute the mean
    #               of the feature and subtract it from the dataset,
    #               storing the mean value in mu. Next, compute the
    #               standard deviation of each feature and divide
    #               each feature by it's standard deviation, storing
    #               the standard deviation in sigma.
    #
    #               Note that X is a matrix where each column is a
    #               feature and each row is an example. You need
    #               to perform the normalization separately for
    #               each feature.
    #
    # Hint: You might find the 'mean' and 'std' functions useful.
    #
    for i in range(X.shape[1]):
        mu[0, i] = np.mean(X[:, i])
        sigma[0, i] = np.std(X[:, i])
        X_norm[:, i] = X[:, i] - mu[0, i]
        X_norm[:, i] = X_norm[:, i] / sigma[0, i]
    # ============================================================

    return X_norm, mu, sigma
