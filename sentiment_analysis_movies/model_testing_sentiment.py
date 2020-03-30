# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:35:41 2020

@author: Piter
"""
import re
import nltk
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Downloading list of irrelevant words and stemmer tool
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



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
        if cnt % 1000 == 0:
            print(cnt)
        cnt += 1
    return my_dataset



train_data = pd.read_csv('train.tsv', delimiter='\t')
test_data = pd.read_csv('test.tsv', delimiter='\t')

train_data.info()
train_data.describe()
train_data.hist(bins=50, figsize=(20,15))

train_data['Sentiment'].value_counts()



X = train_data[['SentenceId', 'Phrase']]
y = train_data['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)


X_train = cleanup_dataset(X_train, 'Phrase')

X_train.to_pickle('X_train')


cv = CountVectorizer(min_df=2)
cv.fit_transform(my_dataset[column_name]).toarray()


