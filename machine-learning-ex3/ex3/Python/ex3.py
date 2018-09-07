# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 00:29:55 2018

@author: dell
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math


mat = scipy.io.loadmat('ex3data1.mat')
X=mat['X']
Y=mat['y']

input_layer_size  = 400;  
num_labels = 10; 
print('Loading and Visualizing Data ...\n')
m = np.size(X, 0);

import random
print(random.sample(range(1, 100), 3))

sel = X[random.sample(range(1,5000), 100), :]


#reshaped = np.reshape(sel[44,:], (20,20))
#reshaped = np.roll(sel[2,:], 1, axis=0)
W = np.zeros((100,20,20))
for i in range(1,100):
    W[i] = sel[i,:].reshape(20,20)
    W[i]=W[i].transpose()
    plt.imshow(W[i,:], cmap='gray')
    plt.show()
#sel.T.reshape(100,-1)
def sigmoid(z):
    g = np.divide(1.0 , np.add(1.0, np.exp(-1*z)))
    return g
m = 5000
y=Y
theta = np.random.rand(400,10)
def lrCostFunction(theta, X, y, lamda):
    J = -(1/m)*np.sum(np.add(np.multiply(y,np.log10(sigmoid(X.dot(theta)))) , np.multiply(np.subtract(1,y),np.log10(np.subtract(1,sigmoid(X.dot(theta)))))))
    J = np.add(J , np.multiply(lamda/(2.0*m),theta[2:,:].T.dot(theta[2:,:])))
    reg = np.append(np.zeros(np.size(theta,1)),np.multiply(lamda/(1.0*m),theta[2:,:]))
    grad = (1/m)*(X.dot(np.subtract(sigmoid(X.dot(theta))-y))) + reg
    
    return J,grad

J,grad = lrCostFunction(theta,X,y,0.01)

