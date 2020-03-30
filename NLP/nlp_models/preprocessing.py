# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:09:08 2020

@author: Piter
"""
import re
import nltk
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

# Downloading list of irrelevant words and stemmer tool
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


cv = CountVectorizer(min_df=2)


# =================================================================
# Importing dataset
# =================================================================
def import_dataset(dataset_name, extension):
    dataset = pd.read_csv('../datasets/' + dataset_name + '.' + extension,
                          delimiter='\t',
                          quoting=3)  # ignoring double quotes
    return dataset


# =================================================================
# Cleaning the text
# =================================================================
def review_cleanup(review):
    # Removing unnecessary signs
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()

    # Removing irrelevant words and stemming
    review = review.split()
    ps = PorterStemmer()
    stopwrd = set(stopwords.words('english'))
    stopwrd = stopwrd.difference(['not', 'don\'t', 'don', 'no'])
    review = [ps.stem(word) for word in review if word not in stopwrd]
    review = ' '.join(review)
    return review


def cleanup_dataset(my_dataset, column_name):
    my_dataset[column_name + '_Clean'] = np.nan
    cnt = 0
    for idx, row in my_dataset.iterrows():
        my_dataset.loc[idx, column_name + '_Clean'] = review_cleanup(my_dataset.loc[idx, column_name])
        if cnt % 1000 == 0 and cnt > 0:
            print(str(cnt) + ' records cleaned')
        cnt += 1
    return my_dataset[column_name + '_Clean']


# =================================================================
# Building sparse matrix
# =================================================================
def get_dataset_sparse_matrix(my_dataset, column_name):
    return cv.fit_transform(my_dataset[column_name]).toarray()


def get_string_sparse_matrix(my_string):
    return cv.transform([review_cleanup(my_string)]).toarray()


