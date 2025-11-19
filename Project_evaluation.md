### This is a guide for reviewers to find relevant information and files. 

* Problem description
    * 0 points: Problem is not described
    * 1 point: Problem is described in README birefly without much details
    * (README.md) **2 points: Problem is described in README with enough context, so it's clear what the problem is and how the solution
will be used**
* EDA
    * 0 points: No EDA
    * 1 point: Basic EDA (looking at min-max values, checking for missing values)
    * (README.md for details, 1_EDA.ipynb for code and plots) **2 points: Extensive EDA (ranges of values, missing values, analysis of target variable, feature importance analysis)**
      For images: analyzing the content of the images.
      For texts: frequent words, word clouds, etc
* Model training
    * 0 points: No model training
    * 1 point: Trained only one model, no parameter tuning
    * 2 points: Trained multiple models (linear and tree-based).
      For neural networks: tried multiple variations - with dropout or without, with extra inner layers or without 
    * (README.md for details, 2_modeling_curated.ipynb and 3_modeling_full.ipynb for code) **3 points: Trained multiple models and tuned their parameters.**
      For neural networks: same as previous, but also with tuning: adjusting learning rate, dropout rate, size of the inner layer, etc.
* Exporting notebook to script
    * 0 points: No script for training a model
    * (train_final.py) **1 point: The logic for training the model is exported to a separate script**
* Reproducibility
    * 0 points: Not possitble to execute the notebook and the training script. Data is missing or it's not easiliy accessible
    * (You can run `> python train_final.py` and check out the notebooks. Data is in "fhvhv_tripdata_2019-02_subset.parquet".)
      **1 point: It's possible to re-execute the notebook and the training script without errors. The dataset is committed in the project repository or there are clear instructions on how to download the data**
* Model deployment
    * 0 points: Model is not deployed
    * (see README.md for how to run in docker) **1 point: Model is deployed (with Flask, BentoML or a similar framework)**
* Dependency and enviroment management
    * 0 points: No dependency management
    * 1 point: Provided a file with dependencies (requirements.txt, pipfile, bentofile.yaml with dependencies, etc)
    * (see README.md for how to do `uv sync --frozen`) **2 points: Provided a file with dependencies and used virtual environment. README says how to install the dependencies and how to
activate the env**
* Containerization
    * 0 points: No containerization
    * 1 point: Dockerfile is provided or a tool that creates a docker image is used (e.g. BentoML)
    * (see README.md for how to run in docker) **2 points: The application is containerized and the README describes how to build a container and how to run it**
* Cloud deployment
    * (I did not do this.) **0 points: No deployment to the cloud**
    * 1 point: Docs describe clearly (with code) how to deploy the service to cloud or kubernetes cluster (local or remote)
    * 2 points: There's code for deployment to cloud or kubernetes cluster (local or remote). There's a URL for testing - or video/screenshot of testing it

Total max 16 points
