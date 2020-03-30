# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:44:13 2020

@author: Piter
"""

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin


class ModelWrapper:

    def __init__(self, model=None, pipeline=Pipeline([('empty', None)])):
        self._model = model
        self._pipeline = pipeline
        self._param_search = None

    def fit(self, data, labels):
        tr_data = self._pipeline.fit_transform(data)
        self._model.fit(tr_data, labels)

    def run_grid_search(self, data, labels, param_grid, substitute_model=True):
        tr_data = self._pipeline.fit_transform(data)
        self._param_search = GridSearchCV(self._model, param_grid, cv=5,
                                          scoring='neg_mean_squared_error')
        self._param_search.fit(tr_data, labels)
        print(self._param_search.cv)
        print(self._param_search.best_params_)
        print(self._param_search.best_score_)
        if substitute_model:
            self._model = self._param_search.best_estimator_

    def run_random_search(self, data, labels, param_dist, substitute_model=True):
        tr_data = self._pipeline.fit_transform(data)
        self._param_search = RandomizedSearchCV(self._model, param_distributions=param_dist,
                                                # n_iter=n_iterations,
                                                n_jobs=-1,
                                                verbose=5,
                                                scoring='neg_mean_squared_error')
        self._param_search.fit(tr_data, labels)
        print(self._param_search.cv)
        print(self._param_search.best_params_)
        print(self._param_search.best_score_)
        if substitute_model:
            self._model = self._param_search.best_estimator_

    def predict(self, data):
        tr_data = self._pipeline.transform(data)
        return self.model.predict(tr_data)

    def dump_model(self, filename):
        joblib.dump(self._pipeline, filename + '_pipe.pkl')
        joblib.dump(self._model, filename + '_model.pkl')

    def load_model(self, filename):
        self._pipeline = joblib.load(filename + '_pipe.pkl')
        self._model = joblib.load(filename + '_model.pkl')

    @property
    def param_search(self):
        return self._param_search

    @param_search.setter
    def param_search(self, param_search):
        self._param_search = param_search

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model


class CustomTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, transformer_func, params=None):
        self._params = params
        self._transformer_func = transformer_func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self._transformer_func(X, *self._params)

