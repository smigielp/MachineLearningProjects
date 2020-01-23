# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 13:20:53 2019

@author: Piter
"""

import data_source, data_utils, pre_analysis_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

data = data_source.load_housing_data()

data.info()
data["ocean_proximity"].value_counts()
datadesc = data.describe()
data.hist(bins=50, figsize=(20,15))

data_with_id = data.reset_index()

print(data.loc[[0, 1],["longitude", "latitude"]])



import data_source, data_utils, pre_analysis_utils
x = data.iloc[:, 0:]

train, test = data_utils.rand_split_train_test(x, 0.2)

encoder = data_utils.NumericImputer(column_names=['total_bedrooms', 'population'])

train = encoder.fit_transform(train)

train = data_utils.transform_data(train, nan_columns=['total_bedrooms'], label_columns=['ocean_proximity'])

x.info()

pre_analysis_utils.plotData(data, 'longitude', 'latitude', data['population'], 'median_house_value', 'Population')

pre_analysis_utils.getCorrespondenceMatrix(data, ["median_house_value", "median_income", 
                                                  "total_rooms", "housing_median_age"])

data.hist(column='median_income', bins=50)

data_utils.categorize_num_data(data, column_to_cat='median_income', cat_limit=1.5, upper_limit_merge=5.0)
data.hist(column='median_income_cat', bins=50)


train, test = data_utils.stratify_split_train_test(data, 'median_income_cat', 0.2)


#===================================================================================

arr = [[0,'pirer', 'a', 100], [1, 'seba', 'b', np.nan], [np.nan, 'anna', 'c', 300], [9, 'grzegorz', 'd', 400]]
test = pd.DataFrame(arr, columns=['rownum', 'name', 'letter', 'value'])

imputer = data_utils.CustomImputer(column_names=['rownum', 'value'])

test[['rownum', 'value']] = imputer.fit_transform(test)


test.reset_index(drop=True, inplace=True)
test.set_index(1, inplace=True)

test.interpolate(method='polynomial', order=2)

data_utils.fillnaMedian(test, [0, 3])


X, y = data_utils.hash_split_train_test(test, 0.2, 0)


