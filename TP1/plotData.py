import matplotlib.pyplot as plt


def plotData(X, y):
    """
    plots the data points and gives the figure axes labels of
    population and profit.
    """

    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the training data into a figure using the
    #               "figure" and "plot" commands. Set the axes labels using
    #               the "xlabel" and "ylabel" commands. Assume the
    #               population and revenue data have been passed in
    #               as the x and y arguments of this function.
    #
    # Hint: You can use the 'ro' option with plt.plot to have the markers
    #       appear as red crosses. Furthermore, you can make the
    #       markers larger by using plt.plot(..., 'r0', markersize=10);

    fig = plt.figure()  # open a new figure window
    plt.plot(X, y, 'rx', markersize=10)
    plt.grid(True)  # Always plot.grid true!
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.show()

# ============================================================
