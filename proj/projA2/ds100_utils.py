from hashlib import md5
from itertools import islice
from pathlib import Path
import requests
import time 

from joblib import dump
import numpy as np
import pandas as pd

#### LAB UTILITIES ####

def download_lab3_data():
    data_dir = 'data'
    data_url = 'http://www.ds100.org/sp24/resources/assets/datasets/lab03_data_sp24.zip'
    file_name = 'lab03_data_sp24.zip'
    return fetch_and_cache(data_url=data_url, file=file_name, data_dir=data_dir)
    

#### PROJECT UTILITIES ####

def run_linear_regression_test(
    final_model, 
    process_data_fm, 
    threshold, 
    train_data_path, 
    test_data_path, 
    is_test=False, 
    is_ranking=False,
    return_predictions=False
):
    def rmse(predicted, actual):
        return np.sqrt(np.mean((actual - predicted)**2))

    training_data = pd.read_csv(train_data_path, index_col='Unnamed: 0')
    X_train, y_train = process_data_fm(training_data)
    if is_test:
        test_data = pd.read_csv(test_data_path, index_col='Unnamed: 0')
        X_test = process_data_fm(test_data, is_test_set = True)
        assert len(test_data) == len(X_test), 'You may not remove data points from the test set!'

    final_model.fit(X_train, y_train)
    if is_test:
        return final_model.predict(X_test)
    else:
        y_predicted = final_model.predict(X_train)
        loss = rmse(y_predicted, y_train)
        if is_ranking:
            print('Your RMSE loss is: {}'.format(loss))
            return loss
        return loss < threshold
    

def run_linear_regression_test_optim(
    final_model, 
    process_data_fm, 
    train_data_path, 
    test_data_path, 
    is_test=False, 
    is_ranking=False,
    return_predictions=False
):
    def rmse(predicted, actual):
        return np.sqrt(np.mean((actual - predicted)**2))

    training_data = pd.read_csv(train_data_path, index_col='Unnamed: 0')
    X_train, y_train = process_data_fm(training_data)
    if is_test:
        test_data = pd.read_csv(test_data_path, index_col='Unnamed: 0')
        X_test = process_data_fm(test_data, is_test_set = True)
        assert len(test_data) == len(X_test), 'You may not remove data points from the test set!'

    final_model.fit(X_train, y_train)
    if is_test:
        return final_model.predict(X_test)
    else:
        y_predicted = final_model.predict(X_train)
        loss = rmse(y_predicted, y_train)
        if is_ranking:
            print('Your RMSE loss is: {}'.format(loss))
            return loss
        fn = (lambda threshold: loss < threshold)
        fn.loss = loss
        fn.signature = (process_data_fm, train_data_path, test_data_path)
        return fn