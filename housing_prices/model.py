# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:18:53 2019

@author: Piter
"""

import data_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = data_utils.load_housing_data()

data.info()
data["ocean_proximity"].value_counts()
datadesc = data.describe()
data.hist(bins=50, figsize=(20,15))

data_with_id = data.reset_index()

print(data.loc[[0, 1],["longitude", "latitude"]])

arr = [[0,'pirer', 'a'], [1, 'seba', 'b'], [np.nan, 'anna', 'c'], [9, 'grzegorz', 'd']]
test = pd.DataFrame(arr)
test.reset_index(drop=True, inplace=True)
test.set_index(1, inplace=True)

test.index

test.loc['seba':, [2]]

test.interpolate(method='polynomial', order=2)


