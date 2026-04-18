import mlflow.sklearn

model_name = 'exp_loan_prediction'
alias = '@champion'

# set tracking uri
mlflow.set_tracking_uri('http://0.0.0.0:5001/')

# Load the first model
loaded_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name + alias}") # using registered model name

# predict on a pandas dataframe
import pandas as pd

data = [[
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                4.98745,
                360.0,
                1.0,
                2.0,
                8.698
            ]]
print(f"Prediction is : {loaded_model.predict(pd.DataFrame(data))}")
