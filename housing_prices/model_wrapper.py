# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:44:13 2020

@author: Piter
"""

from sklearn.externals import joblib


class ModelWrapper:
    
    def __init__(self, model):
        self.model = model
        
    def fit(self, data):
        self.model.fit(data)
        
    def predict(self, data):
        return self.model.predict(data)
    
    def dump_model(model, filename):
        joblib.dump(model, filename)
    
    def load_model(filename):
        return joblib.load(filename)
    