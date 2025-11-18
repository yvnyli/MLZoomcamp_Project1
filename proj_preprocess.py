
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import gc
import sys

#================ choose filenames here ================#

parquetfn = "data/fhvhv_tripdata_2019-02.parquet"
pklfn = "cleaned_df_features_2019-02.pkl"

#=======================================================#




def load_preprocess(parquetfn, pklfn):

    print("loading data: ",parquetfn)

    print("resulting dataframe will be saved in: ", pklfn)

    df = pd.read_parquet(parquetfn)


    df = df_preprocess(df)


    # a small set of features that should be easy to handle (no huge one-hot)
    # all numerical features are easy
    numerics = ["wait_time_sec_log1p","trip_miles_log1p","trip_time_log1p"]
    # short list of categoricals
    curated_cat = ["hvfhs_license_num","wav_request_flag","shared_flag_or","day_of_week","hour_of_day"]
    # other categoricals
    other_cat = ["PULocationID","DOLocationID"]




    with open(pklfn, "wb") as f:
        pickle.dump((df, numerics, curated_cat, other_cat), f)
    
    return df
    
    
    
    
    
def df_preprocess(df):
    ### support translating from format in parquet to what's expected by model/dv
    ### works with or without 'base_passenger_fare'
    col_schema = ['DOLocationID',
                 'PULocationID',
                 'access_a_ride_flag',
                 'airport_fee',
                 'bcf',
                 'congestion_surcharge',
                 'dispatching_base_num',
                 'driver_pay',
                 'dropoff_datetime',
                 'hvfhs_license_num',
                 'on_scene_datetime',
                 'originating_base_num',
                 'pickup_datetime',
                 'request_datetime',
                 'sales_tax',
                 'shared_match_flag',
                 'shared_request_flag',
                 'tips',
                 'tolls',
                 'trip_miles',
                 'trip_time',
                 'wav_match_flag',
                 'wav_request_flag']

    if (set(df.columns) != set(col_schema)) & \
        (set(df.columns) != set(col_schema+['base_passenger_fare'])):
        print("Error: Data does not match schema.", file=sys.stderr)
        sys.exit(1)



    # drop extra columns
    df = df.drop(columns=['dispatching_base_num','originating_base_num','tolls','bcf','sales_tax',
                                'congestion_surcharge','airport_fee','tips','driver_pay',
                                'access_a_ride_flag','wav_match_flag'])
    gc.collect()

    # use wellkown labels for license num
    df['hvfhs_license_num'] = df['hvfhs_license_num'].map(
        {'HV0002':'Juno' ,'HV0003':'Uber', 'HV0004':'Via', 'HV0005':'Lyft'})


    # fill request_datetime
    tdiff = df['pickup_datetime'] - df['request_datetime']
    df.loc[df['request_datetime'].isnull(), 'request_datetime'] = \
    df['pickup_datetime'][df['request_datetime'].isnull()] - tdiff.mean()

    # fill on_scene_datetime
    tdiff = df['pickup_datetime'] - df['on_scene_datetime']
    df.loc[df['on_scene_datetime'].isnull(), 'on_scene_datetime'] = \
    df['pickup_datetime'][df['on_scene_datetime'].isnull()] - tdiff.mean()

    # combine two flags about shared rides
    srs1 = df['shared_request_flag'].map({'Y':True ,'N':False})
    srs2 = df['shared_match_flag'].map({'Y':True ,'N':False})
    df = df.drop(columns=['shared_request_flag','shared_match_flag'])
    gc.collect()
    srs3 = srs1 | srs2
    df = df.assign(shared_flag_or=srs3)

    # turn Y/N into actual binary values
    df['wav_request_flag'] = df['wav_request_flag'].map({'Y':True ,'N':False})



    # apply log transform to right-skewed data
    oldn = df.shape[0]
    df=df[df["trip_miles"].isnull() |(df["trip_miles"] > -1)].copy()
    if df.shape[0] == 0:
        print("Error: Table is empty after removing invalid trip_miles values.", file=sys.stderr)
        sys.exit(1)
    if df.shape[0] < oldn:
        print("Warning: Removed ",(oldn-df.shape[0]), "row(s) due to invalid trip_miles value(s).")
    df= df.assign(trip_miles_log1p=np.log1p(df['trip_miles']))
    
    
    oldn = df.shape[0]
    df=df[df["trip_time"].isnull() |(df["trip_time"] > -1)].copy()
    if df.shape[0] == 0:
        print("Error: Table is empty after removing invalid trip_time values.", file=sys.stderr)
        sys.exit(1)
    if df.shape[0] < oldn:
        print("Warning: Removed ",(oldn-df.shape[0]), "row(s) due to invalid trip_time value(s).")
    df= df.assign(trip_time_log1p=np.log1p(df['trip_time']))

    # turn these into categorical labels
    df["DOLocationID"] = df["DOLocationID"].astype(str)
    df["PULocationID"] = df["PULocationID"].astype(str)

    # again drop the unwanted columns
    df = df.drop(columns=["on_scene_datetime","dropoff_datetime"])
    gc.collect()

    # engineer new features
    oldn = df.shape[0]
    df=df[(df['pickup_datetime']-df['request_datetime']).isnull() | 
    ((df['pickup_datetime']-df['request_datetime']).dt.total_seconds() > -1)].copy()
    if df.shape[0] == 0:
        print("Error: Table is empty after removing invalid wait_time_sec values.", file=sys.stderr)
        sys.exit(1)
    if df.shape[0] < oldn:
        print("Warning: Removed ",(oldn-df.shape[0]), "row(s) due to invalid wait_time_sec value(s).")
    df = df.assign(wait_time_sec_log1p=np.log1p((df['pickup_datetime']-df['request_datetime']).dt.total_seconds()))
    df = df.drop(columns=["pickup_datetime"])
    gc.collect()

    # day of the week
    df = df.assign(day_of_week=df['request_datetime'].dt.day_name())
    # hour of the day
    df = df.assign(hour_of_day=df['request_datetime'].dt.hour)
    df = df.drop(columns=["request_datetime"])
    gc.collect()
    df["hour_of_day"] = df["hour_of_day"].astype(str)
    
    return df




if __name__ == '__main__':
    
    df = load_preprocess(parquetfn, pklfn)  # load dataset and train encoder