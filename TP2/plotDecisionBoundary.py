import numpy as np
from matplotlib import pyplot as plt
from plotData import plotData

def plotDecisionBoundary(theta, X, y, Lambda):
    """
    Plots the data points X and y into a new figure with the decision boundary 
    defined by theta     
      PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
      positive examples and o for the negative examples. X is assumed to be
      a either
      1) Mx3 matrix, where the first column is an all-ones column for the
         intercept.
      2) MxN, N>3 matrix, where the first column is all-ones
    """

    # Plot Data
    plt.figure()
    plotData(X[:,1:], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:, 2]),  max(X[:, 2])])

        # Calculate the decision boundary line
        plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

    else:

        xvals = np.linspace(-1,1.5,50)
        yvals = np.linspace(-1,1.5,50)
        zvals = np.zeros((len(xvals),len(yvals)))
        for i in range(len(xvals)):
            for j in range(len(yvals)):
                myfeaturesij = mapFeature(np.array([xvals[i]]),np.array([yvals[j]]))
                zvals[i][j] = np.dot(theta.flatten(),myfeaturesij.T)
        zvals = zvals.transpose()
    
        u, v = np.meshgrid( xvals, yvals )
        mycontour = plt.contour( xvals, yvals, zvals, [0])
        #Kind of a hacky way to display a text on top of the decision boundary
        myfmt = { 0:'Lambda = %d'% Lambda}
        plt.clabel(mycontour, inline=1, fontsize=15, fmt=myfmt)
        plt.title("Decision Boundary")
        plt.show()

def mapFeature(x1col, x2col, degree=6):
    """
    Feature mapping function to polynomial features

    MAPFEATURE(X, degree) maps the two input features
    to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    """
    """ 
    Function that takes in a column of n- x1's, a column of n- x2s, and builds
    a n- x 28-dim matrix of featuers as described in the homework assignment
    """
    out = np.ones( (x1col.shape[0], 1) )

    for i in range(1, degree+1):
        for j in range(0, i+1):
            term1 = x1col ** (i-j)
            term2 = x2col ** (j)
            term  = (term1 * term2).reshape( term1.shape[0], 1 ) 
            out   = np.hstack(( out, term ))
    return out    

        
