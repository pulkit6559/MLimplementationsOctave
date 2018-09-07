# -*- coding: utf-8 -*-
import numpy as np
import math
import time
X = np.ndfromtxt('images.csv', delimiter=',')
y = np.ndfromtxt("labels.csv", delimiter=',', dtype=np.int8)
img_size = X.shape[1]

# filter out only 0 and 1 and split data
ind = np.logical_or(y == 1, y == 0)
X = X[ind, :]
y = y[ind]

num_train = int(len(y) * 0.8)
X_train = X[0:num_train, :]
X_test = X[num_train:-1,:]
y_train = y[0:num_train]
y_test = y[num_train:-1]