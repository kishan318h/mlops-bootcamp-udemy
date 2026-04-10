## Components of mlflow
1. mlflow tracking:
    - record and query experiments: code, data, config, and results
2. mlflow projects:
    - package data science code in a format to reproduce runs on any platform
3. mlflow models:
    - deploy machine learning models in diverse serving environments
4. model registry:
    - store, annotate, discover and manage models in a central repository


## Features of mlflow
- flexibility to conduct and track numerous experiments before moving a model to production
- it records model evaluation metrics like RMSE and AUC, while also maintaining a log of hyperparameters employed during model development
- it facilitates the storage of the trained model in conjunction with its optimal hyperparameters
- empowers used to seamlessly deploy ML models to production servers or cloud environment
- it allows for the monitoring of models in both staging and production , ensuring that all the team members stay informed
- it is compatible with various popular ML libraries
- it supports multiple programming languages, accessible via REST API and Command Line Interface (CLI)


## Use cases of MLflow
- **Comparing different models:** using the MLflow UI we can compare multiple ML models side by side, along with their metrics and parameter settings
- **Cyclic model deployment:** to push models reliably to the production environment the changes in Data, Requirements, Model's performance. MLflow helps in tracking the models effectively with its metadata
- **Multiple Dependencies:** maintaining the dependencies in a large project with model
- **Working with Large Data Science Team:** To track the Model metadata by extracting the work from other team members by creating the queries


## Getting started with MLflow
### Setup virtual environment:
    - Create virtual environment with Conda: `conda create -n mlflow-venv python=3.12`
    - Activate the environment: `conda activate mlflow-venv`
    - Install mlflow: `pip install mlflow`
    - Open mlflow UI: type the command `mlflow ui` in the terminal and paste the link in a browser
    - if there is other service running on the port then identify it and kill the service
        - `sudo lsof -i tcp:5000` - get a list of services and PID running
        - `kill -15 <PID>`
        