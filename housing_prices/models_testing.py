# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:18:53 2019

@author: Piter
"""

import data_source, data_utils, model_wrapper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV


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
lin_reg = LinearRegression()
lin_reg.fit(train_data, train_labels)

# Verifying results
lin_reg_predictions = lin_reg.predict(some_data_prepared)
print('Prognozy: ', lin_reg_predictions)
print('Etykiety: ', list(some_labels))
lin_mse = mean_squared_error(lin_reg_predictions, some_labels)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)


#==============================================================================
# Training regression model with DecisionTree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_data, train_labels)

# Verifying results
tree_reg_predictions = tree_reg.predict(some_data_prepared)
print('Prognozy: ', tree_reg_predictions)
print('Etykiety: ', list(some_labels))
tree_mse = mean_squared_error(tree_reg_predictions, some_labels)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

# K-folds cross validation (10 folds)
scores = cross_val_score(tree_reg, train_data, train_labels, 
                         scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)
print('Wynik: ', tree_rmse_scores)
print('Średnia: ', tree_rmse_scores.mean())
print('Odchylenie std: ', tree_rmse_scores.std())


#==============================================================================
# Training regression model with RandomForestRegression
forest_reg = RandomForestRegressor(max_features=8, n_estimators=30)
forest_reg.fit(train_data, train_labels)

# Verifying results
forest_reg_predictions = forest_reg.predict(some_data_prepared)
print('Prognozy: ', forest_reg_predictions)
print('Etykiety: ', list(some_labels))
print('Parametry: ', forest_reg.get_params())
forest_reg_mse = mean_squared_error(forest_reg_predictions, some_labels)
forest_reg_rmse = np.sqrt(forest_reg_mse)
print(forest_reg_rmse)

# K-folds cross validation (10 folds)
scores = cross_val_score(forest_reg, train_data, train_labels, 
                         scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)
print('Wynik: ', tree_rmse_scores)
print('Średnia: ', tree_rmse_scores.mean())
print('Odchylenie std: ', tree_rmse_scores.std())

## WE STILL HAVE OVERFITTING!!!


#==============================================================================
# Grid Search - looking for best hyperparameters of model
param_grid = [
    {'n_estimators': [2, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, 
                           scoring='neg_mean_squared_error')

grid_search.fit(train_data, train_labels)
print(grid_search.best_params_)
print(np.sqrt(-grid_search.best_score_))

#==============================================================================
# Models serialization
joblib.dump(lin_reg, 'linear_regression_20190108.pkl')

joblib.dump(tree_reg, 'decision_tree_regression_20190108.pkl')

joblib.dump(lin_reg, 'random_forest_regression_20190108.pkl')

my_model = model_wrapper.ModelWrapper(tree_reg)

model_wrapper.ModelWrapper.dump_model(my_model, 'tree_reg_model_wrapped.pkl')

print(my_model.predict([some_data_prepared[0]]))
my_model2 = model_wrapper.ModelWrapper.load_model('tree_reg_model_wrapped.pkl')
print(my_model2.predict([some_data_prepared[0]]))

#print(lin_reg.predict([some_data_prepared[0]]))
#lin_reg2 = joblib.load('linear_regression_20190108.pkl')
#print(lin_reg2.predict([some_data_prepared[0]]))

