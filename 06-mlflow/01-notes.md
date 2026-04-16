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

### Notes on MLflow tracking
`mlflow.set_tracking_uri()` connects to tracking URI (Uniform Resource Identifier). We can also set the MLFLOW_TRACKING_URI environment variable to have MLflow find a URI from there. In both cases, the URI can either be a HTTP/HTTPS URI for a remote server, a database conection string, or a local path to log data to directory. The URI defaults to mlruns.

`mlfow.get_tracking_uri()` returns the current tracking URI.

`mlflow.create_experiment()` creates a new experiment and returns its ID. Runs can be launched under the experiment by passing the experiment ID to mlflow.start_run.

`mlflow.set_experiment()` sets an experiment as active. If the experiment does not exist, creates a new experiment. If we do not specify an experiment in the `mlflow.start_run()`, new runs are launched under this experiment

`mlflow.start_run()` returns the currently active run (if exists). or starts a new run and returns a mlflow.ActiveRun object usable as a context manager for the current run. We don't need to call start_run explicitly: calling one of the logging function with no active run automatically starts a new one.

`mlflow.end_run()` ends the currently active run, if any, taking an optional run status.

`mlflow.log_param()` logs a signle key-value metric. The key and value are both strings. Use `mlflow.log_params()` to log multiple parameters at once.

`mlflow.log_metric()` logs a single key-value metric. The value must always be a number. Mlflow remembers the history of values for each metric. Use `mlflow.log_metrics()` to log multiple metrics at once.

`mlflow.set_tag()` sets a single key-value tag in the currently active run. The key and value both are strings. Use `mlflow.set_tags()` to set multiple tags at once.

`mlflow.log_artifact()` logs a local file or directory as an artifact, optionally taking an artifact_path to place it within the runs's artifact URI. Run artifacts can be organised into directory, so we can place an artifact in a directory this way.

`mlflow.log_artifact()` logs all the files in a given directory as atrifacts.


### View logged features
Mlflow has moved to sqllite logging from file base logging. To see the output run `mlflow ui --backend-store-uri sqlite:///mlflow.db` in the terminal and then open the displayed link.


### MLflow Project file [Documentation](https://mlflow.org/docs/latest/ml/projects/)
MLflow Projects provide a standard format for packaging and sharing reproducible data science code. Based on simple conventions, Projects enable seamless collaboration and automated execution across different environments and platforms.
Every MLflow project consists of 3 elements:
- Project Name: A human readable identifier for the project
- Entry Point: Commands which can be execute within the project
    - Parameter: Inputs with types and default values
    - Commands - What gets executed when the entry point runs
    - Environment - The execution context and dependencies
- Environment: The software environment containing all dependencies needed to run the project. MLflow supports multiple environment types:
    - Virtualenv: python_env.yaml
    - Conda: conda.yaml
    - Docker: Dockerfile
    - System: None

To create environment file
- Create conda.yaml file
- add name and channel
- copy the dependencies from the conda.yaml file in mlflow ui

MLflow project can be used to setup the experiment in other team members computer. If we share the entire folder on GIT.
Executing the MLproject file will setup the project and environment. To run the `MLproject` file, go to the folder location via terminal and run the command `mlflow run . --experiment-name Predict_Loan_Status` (experiment name: Predict_Loan_Status)

To run the project directly from GitHub, execute the following in terminal `mlflow run <github link> --experiment-name <name>`
- if the MLproject is inside a repo then copy the ssh link
- if the MLproject is present inside a folder in report, then the link would be `github ssh link + #folder_name`
- the run will create a new conda environment, activate that environment and run `mlflow ui` to see the run in UI
