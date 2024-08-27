# Single Instance Training for Iris

import datetime
import os
import subprocess
import sys

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd

import lightgbm as lgb

import argparse
import logging

logging.getLogger().setLevel(logging.INFO)

logging.info("Parsing arguments")

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model-dir', 
    dest='model_dir',        
    default=os.getenv('AIP_MODEL_DIR'), 
    type=str, 
    help='Location to export GCS model')
args = parser.parse_args()
logging.info(args)

def get_data():
    # Download data
    logging.info("Downloading data")
    iris = load_iris()
    print(iris.data.shape)

    # split data
    print("Splitting data into test and train")
    x, y = iris.data, iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    # create dataset for lightgbm
    print("creating dataset for LightGBM")
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    
    return lgb_train, lgb_eval

def train_model(lgb_train, lg_eval):
    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': {'multi_error'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'num_class' : 3
    }

    # train lightgbm model
    logging.info('Starting training...')
    model = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_eval)
    
    return model

lgb_train, lgb_eval = get_data()
model = train_model(lgb_train, lgb_eval)

# GCSFuse conversion
gs_prefix = 'gs://'
gcsfuse_prefix = '/gcs/'
if args.model_dir.startswith(gs_prefix):
    args.model_dir = args.model_dir.replace(gs_prefix, gcsfuse_prefix)
    dirpath = os.path.split(args.model_dir)[0]
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
        
# save model to file
logging.info('Saving model...')
model_filename = 'model.txt'
gcs_model_path = os.path.join(args.model_dir, model_filename)
model.save_model(gcs_model_path)
