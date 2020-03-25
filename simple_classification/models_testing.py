# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:50:57 2020

@author: Piter
"""

from scipy.io import loadmat
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

mnist = loadmat('mnist-original.mat')

X, y = mnist['data'], mnist['label'][0]

X = X.transpose()

some_digit_raw = X[36000]
some_digit = X[36000].reshape(28, 28)
plt.imshow(some_digit, cmap=matplotlib.cm.binary, interpolation='nearest')
y[36000]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
unique, counts = np.unique(y_train_5, return_counts=True)
print(dict(zip(unique, counts)))

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit_raw])

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')

