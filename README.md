# Predict ride share price

## Problem description
### Motivation
Ride share services (Uber/Lyft) take great care into setting the price of rides. The process is opaque and mysterious; there are even myths about the companies dynamically charging more depending on tiny differences in addresses or time, ripping off customers and drivers ([ref1](https://www.theguardian.com/technology/2025/jun/25/second-study-finds-uber-used-opaque-algorithm-to-dramatically-boost-profits), [ref2](https://www.cnn.com/2024/04/03/business/dynamic-surge-pricing-nightcap)). Thus, it would be very interesting to figure out what features the companies base pricing on.
### Data
The Taxi and Limousine Commission (TLC) of New York City requests and publish ride share trip data (available on [NYC Open Data](https://data.cityofnewyork.us/Transportation/For-Hire-Vehicles-FHV-Active/8wbx-tsch/about_data)).  
A subset of this data is also on [Kaggle, titled "NYC FHV (Uber/Lyft) Trip Data Expanded (2019-2022)"](https://www.kaggle.com/datasets/jeffsinsel/nyc-fhvhv-data/data), where I downloaded from.
*The entire dataset is too large. For this project, only one month of data (one file) is used and it is linked in this repo.*

This table contains 20,159,102 rows and 24 columns about ride share rides that happened in Febrary 2019.  
It is still a very large dataset. While I enjoyed getting experience on working with large data, **for reproducibility, I have included a smaller data file in the same format**, "data/fhvhv_tripdata_2019-02_subset.parquet".

### ML problem
**Let's predict the cost of a ride: base_passenger_fare, which is the base passenger fare before tolls, tips, taxes, and fees.**
### Application and significance
Once we have a good model, we can  
1. understand how pricing works,
1. predict fare prices,
2. compare our prediction to prices in ride share apps (to see if we are dynamically charged more),
3. compare price changes over the years...

## Artifacts
I worked in several Jupyter notebooks before nailing down a final model for deployment. The notebooks are labeled in order of my work.  
They can be run from head to toe but will take a long time (hours to overnight for XGB training).
- 1_EDA.ipynb, where I ...
  - understood what each column means,
  - browsed for data type, range of values, unique values, and missing values,
  - picked and cleaned features I think are useful,
  - checked distribution plots and applied transform,
  - engineered more features,
  - produced a cleaned dataframe, saved in "cleaned_df_features.pkl", for loading without rerunning the notebook.
- 2_modeling_curated.ipynb, where I ...
  - in the interest of starting small and getting practice, started training models on a smaller set of curated features,
  - trained linear regression and tree-based models, and tuned hyperparameters,
      - For linear regression, I tried both ridge (L2) and elastic net (L1 and L2) regularization, and tuned regularization strength and L1 ratio.
      - For tree-based models, I used XGBoost and tuned max_depth, learning_rate, min_child_weight, subsample, and colsample_bytree.
  - encountered classic RAM and training time issues because the dataset is overly large, and learned a lot.
- 3_modeling_full.ipynb, where I ...
  - unsatisfied with the performance on limited features, and assessing the feasibility to add more complexity, trained the models on all features,
- train_final.py, where I ...
  - selected the best model based on validation performance obtained in 2_ and 3_,
    - by simply checking the RMSE number, choosing the lowest, and jotting down the hyperparams
  - retrained it on full non-testing data,
  - checked performance on testing data to approximate how it will do in production.

The preprocessing workflow and the final model training workflow have been converted to Python scripts (proj_preprocess.py and train_final.py).

For reproducibility, I changed data path to the smaller file, so you can test the scripts quickly.

## Deployment

**Clone**  
Clone this repo (and `cd` into it in cmd).  
"Dockerfile" will make sure you get all dependencies from "pyproject.toml" and "uv.lock".

**Build**  
`docker build -t fastapi-uv .`

**Run**  
`docker run -p 9696:9696 fastapi-uv`

**Test**  
Visit: http://localhost:9696/docs

**Here are some test cases you can paste:**

1. Ground truth: 182.29  
```
{"hvfhs_license_num":"HV0003","dispatching_base_num":"B02764","originating_base_num":"B02764","request_datetime":1550567148000,"on_scene_datetime":1550568539000,"pickup_datetime":1550568671000,"dropoff_datetime":1550571604000,"PULocationID":48,"DOLocationID":132,"trip_miles":27.64,"trip_time":2934,"tolls":5.76,"bcf":4.7,"sales_tax":16.69,"congestion_surcharge":2.75,"airport_fee":null,"tips":18.15,"driver_pay":155.58,"shared_request_flag":"N","shared_match_flag":"N","access_a_ride_flag":"N","wav_request_flag":"N","wav_match_flag":null}
```

2. Ground truth: 20.34  
```
{"hvfhs_license_num":"HV0003","dispatching_base_num":"B02617","originating_base_num":"B02617","request_datetime":1550695843000,"on_scene_datetime":1550695857000,"pickup_datetime":1550696106000,"dropoff_datetime":1550697076000,"PULocationID":116,"DOLocationID":140,"trip_miles":6.57,"trip_time":969,"tolls":0.0,"bcf":0.51,"sales_tax":1.81,"congestion_surcharge":2.75,"airport_fee":null,"tips":0.0,"driver_pay":15.79,"shared_request_flag":"N","shared_match_flag":"N","access_a_ride_flag":"N","wav_request_flag":"N","wav_match_flag":null}
```

3. Ground truth: 20.68  
```
{"hvfhs_license_num":"HV0003","dispatching_base_num":"B02875","originating_base_num":"B02875","request_datetime":1549458548000,"on_scene_datetime":1549458661000,"pickup_datetime":1549458701000,"dropoff_datetime":1549459853000,"PULocationID":249,"DOLocationID":137,"trip_miles":2.37,"trip_time":1152,"tolls":0.0,"bcf":0.52,"sales_tax":1.84,"congestion_surcharge":2.75,"airport_fee":null,"tips":0.0,"driver_pay":12.45,"shared_request_flag":"N","shared_match_flag":"N","access_a_ride_flag":"N","wav_request_flag":"N","wav_match_flag":null}
```

4. Ground truth: 7.47  
```
{"hvfhs_license_num":"HV0005","dispatching_base_num":"B02510","originating_base_num":null,"request_datetime":1549393556000,"on_scene_datetime":null,"pickup_datetime":1549393722000,"dropoff_datetime":1549394059000,"PULocationID":227,"DOLocationID":227,"trip_miles":1.26,"trip_time":299,"tolls":0.0,"bcf":0.19,"sales_tax":0.66,"congestion_surcharge":0.0,"airport_fee":null,"tips":0.0,"driver_pay":5.39,"shared_request_flag":"N","shared_match_flag":"N","access_a_ride_flag":"N","wav_request_flag":"N","wav_match_flag":null}
```

## Cloud deployment
I will do it in the next project.
