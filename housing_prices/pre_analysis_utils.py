# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 12:47:07 2019

@author: Piter
"""

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


def getCorrespondenceMatrix(X, attributes_list):
    corr_matrix = X.corr()
    for attr in attributes_list:
        print(corr_matrix[attr].sort_values(ascending=False))
        print()
    scatter_matrix(X[attributes_list], figsize=(12,8))
    

def plotData(X, x_axis, y_axis, size_value, color_value, label):
    X.plot(kind='scatter', x=x_axis, y=y_axis, alpha=0.4,
           s=size_value/100, label=label, figsize=(10,7),
           c=color_value, cmap=plt.get_cmap('jet'), colorbar=True)
    