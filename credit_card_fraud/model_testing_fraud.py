# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import hashlib
import matplotlib
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve


data = pd.read_csv('creditcard.csv')

'''
data.info()
data.describe()
data['Amount'].hist(bins=50, figsize=(20,15))
data['Class'].value_counts()
'''

'''
def test_set_check(identifier, test_ratio, hash_func):
    return hash_func(np.int64(identifier)).digest()[-1] < 256 * test_ratio

ids = data['Time']
in_test_set = ids.apply(lambda id_: test_set_check(id_, 0.1, hashlib.md5))
train_val_set, test_set = data.loc[~in_test_set], data.loc[in_test_set]
'''

split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in split.split(data, data['Class']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
    strat_train_set.reset_index(drop=True, inplace=True)
    strat_test_set.reset_index(drop=True, inplace=True)
train_val_set, test_set = strat_train_set, strat_test_set

train_val_set['Class'].value_counts()
test_set['Class'].value_counts()


qt = QuantileTransformer(output_distribution='normal', random_state=42)
train_val_set[['Amount']] = qt.fit_transform(train_val_set[['Amount']])


# =============================================================================
# Training and validation


shuffled_indices = np.random.permutation(len(train_val_set))
test_set_size = int(len(train_val_set) * 0.1)
test_indices = shuffled_indices[:test_set_size]
train_indices = shuffled_indices[test_set_size:]
train_set, val_set = train_val_set.iloc[train_indices], train_val_set.iloc[test_indices]


sgd = SGDClassifier(random_state=42)
svc = SVC(kernel='poly', class_weight='balanced', probability=True)
dtc = DecisionTreeClassifier(max_depth=20)
rfc = RandomForestClassifier(n_estimators=100, max_depth=20)


classifier = sgd


# =============================================================================
# Training

X = train_set.drop(['Time', 'Class'], axis=1)
y = train_set['Class']


classifier.fit(X, y)

y_pred = classifier.predict(X)
cv = confusion_matrix(y, y_pred)
print(cv)


# =============================================================================
# Validation

X_val = val_set.drop(['Time', 'Class'], axis=1)
y_val = val_set['Class']


y_val_pred = classifier.predict(X_val)
cv1 = confusion_matrix(y_val, y_val_pred)
print(cv1)


# =============================================================================
# Testing

test_set[['Amount']] = qt.transform(test_set[['Amount']])
X_test = test_set.drop(['Time', 'Class'], axis=1)
y_test = test_set['Class']


y_test_pred = classifier.predict(X_test)
cv_test = confusion_matrix(y_test, y_test_pred)
print(cv_test)


# =============================================================================
# Cross validation

y_scores = cross_val_score(classifier, X, y, cv=3, scoring='accuracy')
y_decision = cross_val_predict(classifier, X, y, cv=3, method='decision_function')  # Only for Linear Model


# =============================================================================
# Classifier assessment

print(roc_auc_score(y_test, y_test_pred))
print("Precision: ", precision_score(y_test, y_test_pred))
print("Recall:    ", recall_score(y_test, y_test_pred))


fpr, tpr, thresholds = roc_curve(y, y_decision)

plt.plot(fpr, tpr, linewidth=2, label=None)
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('Odsetek fałszywie pozytywnych')
plt.ylabel('Odsetek prawdziwie pozytywnych')
plt.show()

precisions, recalls, thresholds = precision_recall_curve(y, y_decision)

plt.plot(thresholds, precisions[:-1], 'b--', label='Precyzja')
plt.plot(thresholds, recalls[:-1], 'g-', label='Pelnosc')
plt.xlabel('Prog')
plt.legent(loc='right')
plt.ylim([0, 1])
plt.show()


# RandomForestClassifier on test data:
#[[28359     1]
# [    9    38]]





