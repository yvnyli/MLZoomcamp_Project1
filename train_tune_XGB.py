from proj_preprocess import load_preprocess

import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import gc
from pprint import pprint

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

#================ choose filenames here ================#

parquetfn = "data/fhvhv_tripdata_2019-02.parquet"
pklfn = "cleaned_df_features_2019-02.pkl"

bestfn = "script_train_tune_XGB_best_xgb.pkl"

#=======================================================#





def get_X_y_dv(df):
    ### expect output from load_preproces
    ### i.e. a transformed df containing 'base_passenger_fare'
    dv_full = DictVectorizer(sparse=True)
    
    y = df.base_passenger_fare.values

    X = dv_full.fit_transform(df.drop(columns='base_passenger_fare').to_dict(orient='records'))
    
    return X,y
    




p_d_XGB = {
    'max_depth': [5,8,11,16],
    'learning_rate': [0.3, 0.1, 0.05],
    'min_child_weight': [8, 25, 50, 100],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.3, 0.6, 0.8]
}



def train_tune_XGB(X,y,param_dist=p_d_XGB):

    print("train_tune_XGB: For illustration purposes, we will train on a smaller subset of the data.")
    # -------------------------------
    # 3. XGBoost: randomized search
    # -------------------------------

    # portion out some data from _train for early stopping
    X_temp, X_stop_xgb, y_temp, y_stop_xgb = train_test_split(
        X, y, test_size=0.00005, random_state=42)
    # use fewer data for training for speed
    X_temp, X_train_xgb, y_temp, y_train_xgb = train_test_split(
        X_temp, y_temp, test_size=0.0001, random_state=42)
        
        
    del X_temp
    del y_temp
    gc.collect()

    print(f"eval (early stopping) on {y_stop_xgb.shape[0]:d} rows")
    print(f"train (CV tuning) on {y_train_xgb.shape[0]:d} rows")

    # Define parameter distribution
    # param_dist = {
        # 'max_depth': [5,8,11,16],
        # 'learning_rate': [0.3, 0.1, 0.05],
        # 'min_child_weight': [8, 25, 50, 100],
        # 'subsample': [0.8, 1.0],
        # 'colsample_bytree': [0.3, 0.6, 0.8]
    # }

    # Create XGBClassifier
    xgb = XGBRegressor(
        tree_method="hist",
        enable_categorical=True,  # if using pandas categorical dtypes
        n_estimators=200,        # large, rely on early stopping
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=4,
        n_jobs=-1
    )

    fit_params = {
        "eval_set": [(X_stop_xgb, y_stop_xgb)],
        "verbose": False,
    }


    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=5,          # e.g. 50 random trials
        scoring="neg_root_mean_squared_error",
        verbose=4,          # shows progress of the search
        n_jobs=2,
        cv=3   
    )

    print("\n=== XGBoost hyperparameter search ===")
    t0 = time.time()
    search.fit(X_train_xgb, y_train_xgb, **fit_params)
    t1 = time.time()




    print("XGB Best params:", search.best_params_)
    print("XGB Best CV score (RMSE):", -search.best_score_)
    print(f"XGB search time: {(t1 - t0)/60:.2f} minutes")

    best_xgb = search.best_estimator_

        
    # Evaluate best XGB on validation set explicitly:
    y_pred_xgb = best_xgb.predict(X_full_train)
    xgb_train_rmse = root_mean_squared_error(y_train, y_pred_xgb)

    y_pred_xgb = best_xgb.predict(X_full_val)
    xgb_val_rmse = root_mean_squared_error(y_val, y_pred_xgb)
    print(f"XGB training RMSE (best model): {xgb_train_rmse:.4f}")
    print(f"XGB validation RMSE (best model): {xgb_val_rmse:.4f}")
    
    # let's check the results
    results = search.cv_results_
    df_results = pd.DataFrame({
        "mean_fit_time": results["mean_fit_time"],
        "param_subsample": results["param_subsample"],
        "param_min_child_weight": results["param_min_child_weight"],
        "param_max_depth": results["param_max_depth"],
        "param_learning_rate": results["param_learning_rate"],
        "param_colsample_bytree": results["param_colsample_bytree"],
        "mean_CV_RMSE": -results["mean_test_score"],
        "std_test_score": results["std_test_score"],
        "rank": results["rank_test_score"],
    })

    df_results.sort_values("rank",inplace=True)
    
    print("Distribution of mean CV RMSE")
    print(df_results["mean_CV_RMSE"].describe())
    
    print("Top 10 models")
    
    pprint(df_results.loc[0:9,:])
    
    
    return best_xgb
    
    
    
if __name__ == '__main__':

        df = load_preprocess(parquetfn, pklfn)

        print(df.shape)

        print(df.isnull().sum())

        print("\n")

        print(df.dtypes)
        
        X,y = get_X_y_dv()
        
        del df
        gc.collect()
        
        
        X_notest, X_test, y_notest, y_test = train_test_split(
            X, y, test_size=0.001, random_state=42)
        
        print("train_tune_XGB: Reserved ",y.shape[0]," testing observations.")
        
        best_xgb = train_tune_XGB(X_notest, y_notest)
        
        
        y_pred_xgb = best_xgb.predict(X_test)
        xgb_test_rmse = root_mean_squared_error(y_test, y_pred_xgb)
        
        print(f"best XGB test RMSE: {xgb_test_rmse:.4f}")
        
        with open(bestfn, "wb") as f:
            pickle.dump(best_xgb, f)