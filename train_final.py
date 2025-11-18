from proj_preprocess import load_preprocess

import pandas as pd
import numpy as np
import pickle
from pprint import pprint

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

from xgboost import XGBRegressor

#================ choose filenames here ================#

parquetfn = "fhvhv_tripdata_2019-02_subset.parquet"
pklfn = "fhvhv_tripdata_2019-02_subset.pkl"

bestfn = "script_train_final_mdl.pkl"

#=======================================================#



with open ("dv_full.pkl", "rb") as f:
    dv_full = pickle.load(f) # DictVectorizer


def get_X_y_dv(df):
    ### expect output from load_preprocces
    ### i.e. a transformed df containing 'base_passenger_fare'
    
    y = df.base_passenger_fare.values

    X = dv_full.transform(df.drop(columns='base_passenger_fare').to_dict(orient='records'))
    
    return X,y
    



def train_final_XGB(X,y):

   
    
    # first specify the model with winning hyperparameters and final training specs
    mdl = XGBRegressor(
        tree_method="hist",
        enable_categorical=True,  # if using pandas categorical dtypes
        n_estimators=500,        # large, rely on early stopping
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=10,
        n_jobs=-1,
        subsample=1.0, 
        min_child_weight=100, 
        max_depth=8, 
        learning_rate=0.05, 
        colsample_bytree=0.6
    )
    

    # split some data for early stopping
    X_train, X_stop_xgb, y_train, y_stop_xgb = train_test_split(
        X, y, test_size=0.1, random_state=0)

    mdl.fit(X_train, y_train, eval_set=[(X_stop_xgb, y_stop_xgb)])

    with open(bestfn, "wb") as f:
        pickle.dump(mdl, f)
    
    return mdl
    
    
if __name__ == '__main__':

    print("train_final_XGB: For illustration purposes, ",
          "we will train on a smaller subset of the data.") 
    df = load_preprocess(parquetfn, pklfn)

    print("Data size:")
    
    df_train, df_test = train_test_split(df, train_size=0.8, random_state=42)
    
    print("For training: ", df_train.shape)
    print("For testing: ", df_test.shape)
    X_train,y_train = get_X_y_dv(df_train)
    
    mdl = train_final_XGB(X_train,y_train)
    

    X_test,y_test = get_X_y_dv(df_test)

    y_pred = mdl.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, y_pred)

    print(f"Testing RMSE: {test_rmse:.4f}")