# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 01:44:56 2018

@author: dell
"""

import numpy as np
import math
import time
X = np.genfromtxt('images.csv', dtype='float64' , delimiter=',')
y = np.genfromtxt("labels.csv", delimiter=',', dtype=np.int8)
img_size = X.shape[1]

ind = np.logical_or(y == 1, y == 0)
X = X[ind, :]
y = y[ind]

num_train = int(len(y) * 0.8)
X_train = X[0:num_train, :]
X_test = X[num_train:-1,:]
y_train = y[0:num_train]
y_test = y[num_train:-1]

max_iter = 10
alpha = 0.01

def h_vec(theta, X):
    return 1 / (1 + np.exp(-np.matmul(X, theta)))

def GD (theta, X_train, y_train, alpha):
    theta -= alpha * np.squeeze(np.matmul(np.reshape(h_vec(theta, X_train) - y_train, [1, -1]), X_train))
    
def train_vec(X_train, y_train, max_iter, alpha):
    theta = np.zeros([img_size])
    for i in range(max_iter):
        GD(theta, X_train, y_train, alpha)       
    return theta

