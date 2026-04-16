import mlflow.sklearn

run_id = "bb0d721b2cb24cdfb908674b2bcd6178" # Use your specific Run ID here
# model_id = 'm-2132c6c27dda4de9b236f8dc7a28d5d7'

# Load the first model
loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/RandomForestClassifier") # using registered model name

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
