# -*- coding: utf-8 -*-
"""
Gradient-descent fitting for linear functions
It will work for non-linear functions as well if the gradient is coded properly
"""
import numpy as np
import matplotlib.pylab as plt

def gradient(x, y, a):
    '''
    Gradient for the linear hypothesis
    Parameters
    ----------
    x : 1d array
        x of the training data.
    y : 1d array
        y of the training data.
    a : 1d array
        the current coefficients to evaluate the gradient
        hypothesis is predi = a[0] + a[1]*xi

    Returns
    -------
    dJ_da : 1d array with the same length as a
        Gradient of the loss function.

    '''
    dJ_da = np.zeros(2)
    error = a[0] + a[1]*x - y
    n = len(x)
    dJ_da[0] = 2/n*np.sum(error)
    dJ_da[1] = 2/n*np.sum(error*x)
    return dJ_da

def Steepest_Descent(x, y, a_in, max_iterations=200, alpha=0.0001, precision = 0.01):
    '''
    General Steepest Descent algorithm for optimization. 
    Parameters
    ----------
    x : 1d array
        x of the training data.
    y : 1d array
        y of the training data.
    a_in : 1d array
        initial guess of coefficients to fit.
        hypothesis is predi=a[0] + a[1]*xi
    max_iterations : integer
        The default is 200.
    alpha : float
        step size in gradient descent. The default is 0.0001.
    precision: float
        convergence criteria, stop when the loss function is smaller than this

    Returns
    -------
    a : 1d array
        optmized coefficients.
    J_history : 1d array
        loss function values over iterations.

    '''
    # make a copy of the coefficient array 
    # to avoid modifying the original input array
    a = a_in.copy()
    # store the loss function at each iteraction
    J_history = []
    for i in range(max_iterations):
        # get the gradient
        dJ_da = gradient(x, y, a)
        # update a following the gradient
        #a[0] = a[0] - dJ_da[0]*alpha
        #a[1] = a[1] - dJ_da[1]*alpha
        a = a - dJ_da*alpha # vector form 
        # loss function
        n = len(x)
        error = a[0] + a[1]*x - y
        J = 1/n * np.sum(error**2)
        #print("a:", a)
        print("J:", J)
        J_history.append(J)
        # exit the loop if the J is small enough
        if J < precision: break
    return a, J_history

# read some test data    
data = np.loadtxt('time-msd.txt')
x = data[:10, 0]
y = data[:10, 1]
# initialize the linear coefficients to [0, 0]
a = np.zeros(2)
a, J_history = Steepest_Descent(x, y, a, max_iterations = 2000)
print("Final coefficients:", a)
#plt.plot(x, y, '^')
#plt.plot(x, a[0] + a[1]*x, label='fit')
plt.plot(J_history, label='loss function')
plt.legend(loc='best')
plt.show()
