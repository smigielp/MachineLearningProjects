# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 12:22:04 2019

@author: Piter
"""

import numpy as np
import pandas as pd
import hashlib
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


#============================================================    
# Custom transformers    
#============================================================

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, families_col_id, population_col_id, rooms_col_id):
        self.families_col_id = families_col_id
        self.population_col_id = population_col_id
        self.rooms_col_id = rooms_col_id
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        rooms_per_family = X[:, self.rooms_col_id] / X[:, self.families_col_id]
        population_per_family = X[:, self.population_col_id] / X[:, self.families_col_id]
        return np.c_[X, rooms_per_family, population_per_family]
    

class NumericImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, strategy='median', column_names=[]):
        self.column_names = column_names
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)
        
    def fit(self, X, y=None):
        self.imputer.fit(X[self.column_names])
        return self
    
    def transform(self, X, y=None):
        if self.column_names == []:
            X = self.imputer.transform(X)
        else:
            X[self.column_names] = self.imputer.transform(X[self.column_names])
        return X
    
    
class ColumnToOneHot(BaseEstimator, TransformerMixin):
    
    def __init__(self, column_name=None, remove_dummy=True):
        self.column_name = column_name
        self.remove_dummy = remove_dummy
        self.encoder = LabelBinarizer()
        
    def fit(self, X, y=None):
        self.encoder.fit(X[self.column_name])
        return self
    
    def transform(self, X, y=None):
        ocean_proximity_one_hot = self.encoder.transform(X[self.column_name]) 
        X = pd.concat([X, pd.DataFrame(ocean_proximity_one_hot)], axis=1)
        X.drop(self.column_name, axis=1, inplace=True)
        if self.remove_dummy == True:
            X.drop(0, axis=1, inplace=True)
        return X
    
#============================================================    
# Pipelines    
#============================================================

class CustomPipeline:
    
    def __init__(self, nan_columns=[], label_columns=[], scale_data=True):
        transformers = []
        if nan_columns != []:
            transformers.append(('imputer', NumericImputer(column_names=nan_columns)))
        for column in label_columns:
            transformers.append(('label_binarizer' + column, ColumnToOneHot(column_name=column)))
        if scale_data:
            transformers.append(('std_scaler', StandardScaler()))
        self.pipeline = Pipeline(transformers)
    
    def fit(self, X):
        self.pipeline.fit(X)
        return self
        
    def transform(self, X):
        return self.pipeline.transform(X)
        
    
# Deprecated function

def transform_data(X, nan_columns=[], label_columns=[], scale_data=True):
    transformers = []
    if nan_columns != []:
        transformers.append(('imputer', NumericImputer(column_names=nan_columns)))
    for column in label_columns:
        transformers.append(('label_binarizer' + column, ColumnToOneHot(column_name=column)))
    if scale_data:
        transformers.append(('std_scaler', StandardScaler()))
    pipeline = Pipeline(transformers)
    return pipeline.fit_transform(X)


#============================================================    
# Train-test dataset splitting    
#============================================================

def rand_split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#------------------------------------------------------------
def test_set_check(identifier, test_ratio, hash_func):
    return hash_func(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def hash_split_train_test(data, test_ratio, id_column, hash_func=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash_func))
    return data.loc[~in_test_set], data.loc[in_test_set]

#------------------------------------------------------------
def categorize_num_data(data, column_to_cat, cat_limit, upper_limit_merge):
    category_column_name = column_to_cat + '_cat'
    data[category_column_name] = np.ceil(data[column_to_cat] / cat_limit)
    data[category_column_name].where(data[category_column_name] < upper_limit_merge, upper_limit_merge, inplace=True)
    

def stratify_split_train_test(data, category_column, test_ratio):
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    for train_index, test_index in split.split(data, data[category_column]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
        strat_train_set.reset_index(drop=True, inplace=True)
        strat_test_set.reset_index(drop=True, inplace=True)
    return strat_train_set, strat_test_set
    

#============================================================
# prognosis-labels colums splitting
#============================================================

def separate_label_data(data, label_column):
    prog_data = data.drop(label_column, axis=1)
    label_data = data[label_column].copy()
    return prog_data, label_data


#============================================================
# Printing Learning Curve
#============================================================
    
def plot_learning_curve(model, X_train, X_val, y_train, y_val):
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), 'r-+', linewidth=2, label='train')
    plt.plot(np.sqrt(val_errors), 'b-', linewidth=2, label='train')
    
    
def plot_learning_curve_raw_data(model, data, label_column):
    X, y = rand_split_train_test(data, 0.2)
    X_train, X_val = separate_label_data(data, label_column)
    y_train, y_val = separate_label_data(data, label_column)
    plot_learning_curve(model, X_train, X_val, y_train, y_val)
    
    