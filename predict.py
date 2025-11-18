import pickle
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any

from proj_preprocess import df_preprocess


import pandas as pd



app = FastAPI(title="ride_share-fare-prediction")


with open("final_model_trained.pkl", "rb") as f:
    mdl = pickle.load(f) # model
    
with open ("dv_full.pkl", "rb") as f:
    dv_full = pickle.load(f) # DictVectorizer
    
with open ("default_nulldf.pkl", "rb") as f:
    nulldf = pickle.load(f) # expected column names and default values
    
def single_dict_to_df(dc):
    # get values from dict
    df = nulldf.copy()
    cols = set(dc.keys()) & set(nulldf.columns)
    df.loc[0, list(cols)] = [dc[c] for c in cols]
    # transform
    df = df_preprocess(df)

def predict_single(trip):
    
    # from dict to df to X with dv
    df = single_dict_to_df((trip)[0, 1])
    X_single = dv_full.transform(df)
    
    fare = mdl.predict(X_single)
    return float(fare)


@app.post("/predict")
def predict(trip: Dict[str, Any]):
    fare = predict_single(trip)

    return {
        "predicted_base_fare": fare
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)