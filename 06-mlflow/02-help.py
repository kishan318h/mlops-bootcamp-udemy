import mlflow
mlflow.set_tracking_uri('https://localhost::5000')

exp_id = mlflow.create_experiment('loan_prediction')

# starting a run
with mlflow.start_run(run_name='DecisiionTreeClassifer') as run:
    mlflow.set_tag('version', '0.0.1')

# end the currently active run
mlflow.end_run()


n_estimator = 100
criterion = 'gini'

mlflow.log_param('n_estimator', n_estimator)
mlflow.log_param('criterion', criterion)

mlflow.log_metric('accuracy', 0.9)

