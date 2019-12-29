# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 21:48:35 2019

@author: Piter
"""

import os
import tarfile
import pandas as pd
from six.moves import urllib

DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml/raw/master/"
HOUSING_PATH = os.path.join("data sets", "housing")
FILE_EXTENSION = "tgz"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing." + FILE_EXTENSION

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH, file_type=FILE_EXTENSION):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    file_path = os.path.join(housing_path, "housing." + file_type)
    urllib.request.urlretrieve(housing_url, file_path)
    if file_type == "tgz":
        housing_tgz = tarfile.open(file_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
        

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

    