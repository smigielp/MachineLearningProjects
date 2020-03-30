from scipy.stats import randint

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

from model_wrapper import ModelWrapper, CustomTransformer

import pandas as pd
import model_wrapper
import preprocessing as prep


dataset = prep.import_dataset('Restaurant_Reviews', 'tsv')


pipeline = Pipeline([
    ('text_transform', model_wrapper.CustomTransformer(prep.cleanup_dataset, ['Review'])),
    ('sparse_matrix', CountVectorizer(min_df=2)) # ,
])


X = dataset[['Review']]
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=46)

classifier = ModelWrapper(SVC(kernel='poly', gamma='scale'), pipeline)


param_grid_svc = [
            {'degree': [2, 3, 4, 5, 6, 7, 8, 9], 'coef0': [1, 2, 3, 4, 5], 'C': [0.01, 0.1, 1, 5, 10, 20, 50, 100]}
        ]
classifier.run_grid_search(X_train, y_train, param_grid_svc)





classifier.fit(X_train, y_train)


classifier.dump_model('svc_20032020')


cls2 = ModelWrapper()
cls2.load_model('svc_20032020')


y_pred = cls2.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('| ' + str(cm[0][0]) + ' | ' + str(cm[0][1] ) + ' |')
print('| ' + str(cm[1][0]) + ' | ' + str(cm[1][1]) + ' |')

review = pd.DataFrame({'Review': ['Big portions and very tasty']})

rev_pred = cls2.predict(review)[0]
print('Thanks for review!' if rev_pred == 1 else 'Sorry, we\'ll try to do better')

