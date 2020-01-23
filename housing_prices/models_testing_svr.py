# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:07:52 2020

@author: Piter
"""


import data_source, data_utils, model_wrapper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


data = data_source.load_housing_data()

# Basic data insight
'''
data.info()
data["ocean_proximity"].value_counts()
data.describe()
data.hist(bins=50, figsize=(20,15))
'''

# Stratified sampling
data_utils.categorize_num_data(data, 'median_income', 1.5, 5.0)
train_data_raw, test_data_raw = data_utils.stratify_split_train_test(data, 'median_income_cat', 0.2)

train_data_raw, train_labels = data_utils.separate_label_data(train_data_raw, 'median_house_value')

# Fixing missing values, performing one-hot-encoding, scaling
my_pipeline = data_utils.CustomPipeline(nan_columns=['total_bedrooms'], 
                                        label_columns=['ocean_proximity'], 
                                        scale_data=True)

train_data = my_pipeline.fit(train_data_raw).transform(train_data_raw)


# Preparing sample for quick pre-verification
some_data = train_data_raw.iloc[:5]
some_labels = train_labels[:5]
some_data_prepared = my_pipeline.transform(some_data)


#==============================================================================
# Training regression model with Linear Regression
sv_reg = SVR(kernel="rbf")
sv_reg.fit(train_data, train_labels)

# Verifying results
sv_reg_predictions = sv_reg.predict(some_data_prepared)
print('Prognozy: ', sv_reg_predictions)
print('Etykiety: ', list(some_labels))
sv_mse = mean_squared_error(sv_reg_predictions, some_labels)
sv_rmse = np.sqrt(sv_mse)
print(sv_rmse)


#==============================================================================
# Random Search Cross Validation
distribution = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
  'kernel': ['rbf']}

sv_reg = SVR()
rand_search = RandomizedSearchCV(sv_reg, distribution, random_state=0)
rand_search.fit(train_data, train_labels)
print(rand_search.best_params_)
print(np.sqrt(-rand_search.best_score_))
rand_search_pred = rand_search.best_estimator_.predict(some_data_prepared)

sv_mse = mean_squared_error(rand_search_pred, some_labels)
sv_rmse = np.sqrt(sv_mse)
print(sv_rmse)

