from matplotlib import pyplot as plt


def plotData(X, y):
    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot data X with different markers according the value in y
    # =============================================================
    pos = X[(y == 1).flatten(), :]
    neg = X[(y == 0).flatten(), :]

    plt.plot(pos[:, 0], pos[:, 1], '+', markersize=7, markeredgecolor='black', markeredgewidth=2)
    plt.plot(neg[:, 0], neg[:, 1], 'o', markersize=7, markeredgecolor='yellow', markeredgewidth=2)

    plt.legend(['Admitted (y=1)', 'Not admittedx (y=0)'], loc='upper right', shadow=True, fontsize='x-large',
               numpoints=1)
    plt.grid()
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

# =============================================================
