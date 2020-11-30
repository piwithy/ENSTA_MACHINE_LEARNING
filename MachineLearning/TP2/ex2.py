#%% Logistic Regression
import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pylab as plt

from plotData import plotData
from plotDecisionBoundary import plotDecisionBoundary
from costFunction import costFunction
from sigmoid import sigmoid

from gradientFunction import gradientFunction
from predict import predict



#%% Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     plotData.py
#     sigmoid.py
#     costFunction.py
#     gradientFunction.py
#     predict.py
#     costFunctionReg.py
#     gradientFunctionReg.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

# Load Data with pandas
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.head()


# set X (training data) and y (target variable)
nbCol = data.shape[1]
X = data.iloc[:,0:nbCol-1]
y = data.iloc[:,nbCol-1:nbCol]

# convert from data frames to numpy arrays
X = np.array(X.values)
y = np.array(y.values)



# %% ==================== Part 1: Plotting ====================
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

plotData(X, y)




#%% ============ Part 2: Compute Cost and Gradient ============
#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to x and X_test
X = np.concatenate((np.ones((m, 1)), X), axis=1)

# Initialize fitting parameters
initial_theta = np.zeros((n + 1,1))

# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y)
print('\n -------------------------- \n')
print('Cost at initial theta (zeros): %f' % cost)
print('Expected cost (approx): 0.693')

grad = gradientFunction(initial_theta, X, y)
print('\n -------------------------- \n')
print('Gradient at initial theta (zeros): ' + str(grad))
print('Expected gradients (approx): -0.1000 -12.0092 -11.2628')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24, 0.2, 0.2]]).T
cost = costFunction(test_theta, X, y)
grad = gradientFunction(test_theta, X, y)

print('\n -------------------------- \n')
print('Cost at test theta: %f' %cost)
print('Expected cost (approx): 0.218')
print('Gradient at test theta:' + str(grad))
print('Expected gradients (approx): 0.043 2.566 2.647')










##%% ============= Part 3: Optimizing using scipy  =============
theta = opt.fmin_tnc(costFunction, initial_theta, gradientFunction, args=(X, y))
theta = theta[0]
cost = costFunction(theta, X, y)


# print(theta to screen
print('\n -------------------------- \n')
print('Cost at theta found by scipy: %f' % cost)
print('Expected cost (approx): 0.203')
print('\n -------------------------- \n')
print('theta:', ["%0.4f" % i for i in theta])
print('Expected theta (approx): -25.161 0.206 0.201');

# Plot Boundary
plotDecisionBoundary(theta, X, y, Lambda=0)






##%%  ============== Part 4: Predict and Accuracies ==============

#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2

prob = sigmoid( np.array([1, 45, 85])@theta )

print('\n -------------------------- \n')
print('For a student with scores 45 and 85, we predict an admission probability of %f' % prob)
print('Expected Proba (approx): 0.776')

# Compute accuracy on our training set
p = predict(theta, X)
accuracy = np.mean(np.double(p == np.squeeze(y))) * 100

print('\n -------------------------- \n')
print('Train Accuracy: %f' % accuracy)
print('Expected accuracy (approx): 89.0%');

plt.show()