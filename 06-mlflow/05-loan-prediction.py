# Import packages
import os
import mlflow
import pandas as pd
import numpy as np
import skops
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics


# load dataset
dataset = pd.read_csv('train.csv')
dataset.columns = [col.strip() for col in dataset.columns]

numerical_cols = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()

categorical_cols.remove('Loan_Status')
categorical_cols.remove('Loan_ID')


# fill nulls in categorical column with Mode
for col in categorical_cols:
    _mode = dataset[col].mode()
    dataset = dataset.fillna({col: _mode})


# fill nulls in numerical columns with Median
for col in numerical_cols:
    _median = dataset[col].median()
    dataset = dataset.fillna({col: _median})


# outlier treatment: 
# setting the value below 5th %tile to value at 5th percentile
# setting the value over 95th %tile to value at 95th percentile
dataset[numerical_cols] = dataset[numerical_cols].clip(
    lower=dataset[numerical_cols].quantile(0.05),
    upper=dataset[numerical_cols].quantile(0.95),
    axis=1
).infer_objects(copy=False)


# log transformation and domain processing
dataset['LoanAmount'] = np.log(dataset['LoanAmount'])
dataset['Total_Income'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['Total_Income'] = np.log(dataset['Total_Income'])


# drop AppicantIncome & CoapplicantIncome
dataset.drop(columns=['ApplicantIncome', 'CoapplicantIncome', 'Loan_ID'], inplace=True)
print('Feature cleaning and enginnering done!')

# X,y
TARGET = 'Loan_Status'
X = dataset.drop(columns=[TARGET])
y = dataset[TARGET]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)
print('Train test split done!')

# label encoding
def label_encoder(df, colnames):
    df_encoded = df.copy()

    le = LabelEncoder()
    for col in colnames:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    return df_encoded


train_features = label_encoder(X_train, categorical_cols)
train_target = y_train.map({'Y': 1, 'N': 0})


# Random forest
print('Traning Random Forest Classifier...')
rf = RandomForestClassifier(random_state=18)
rf_param_grid = {
    'n_estimators': [200, 400, 800],
    'max_depth': [8, 12, 16],
    'criterion': ['gini', 'entropy'],
    'max_leaf_nodes': [50, 80]
}

rf_cv = GridSearchCV(
    estimator= rf,
    param_grid= rf_param_grid,
    cv= 5,
    n_jobs= -1,
    scoring= 'accuracy',
    verbose= 0
)

model_rf = rf_cv.fit(train_features, train_target) # fit random forest
print('Random Forest trained')


# Logistic regression
print('Traning Logistict Regression model...')
lr = LogisticRegression(random_state=18)
lr_param_grid = {
    'C': [100, 10, 1.0, 0.1, 0.01],
    'solver': ['liblinear']
}

lr_cv = GridSearchCV(
    estimator= lr,
    param_grid= lr_param_grid,
    cv= 5,
    n_jobs= -1,
    scoring= 'accuracy',
    verbose= 0
)

model_lr = lr_cv.fit(train_features, train_target) # fit logistic regresion
print('Logistic Regression model trained')


# Decision tree
print('Traning Decision Tree Classifier...')
dt = DecisionTreeClassifier(random_state=18)
dt_param_grid = {
    'max_depth': [3,5,7,9,11],
    'criterion': ['gini', 'entropy']
}

dt_cv = GridSearchCV(
    estimator= dt,
    param_grid= dt_param_grid,
    cv= 5,
    n_jobs= -1,
    scoring= 'accuracy',
    verbose= 0
)

model_dt = dt_cv.fit(train_features, train_target) # fit decision tree
print('Decision Tree model trained')


##### log model and parameters
mlflow.set_experiment("Predict_Loan_Status")

# Model evelaution metrics
def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f'%auc)
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ROC_curve.png")
    # Close plot
    plt.close()
    return(accuracy, f1, auc)


def mlflow_logging(model, X, y, model_name):

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag('run_id', run_id)

        pred = model.predict(X)

        # metrics
        (accuracy, f1, auc) = eval_metrics(y, pred)
        # logging best parameters from grid search
        mlflow.log_params(model.best_params_)
        # log metrics
        mlflow.log_metric('Mean CV score', model.best_score_)
        mlflow.log_metric('Accuracy', accuracy)
        mlflow.log_metric('f1-score', f1)
        mlflow.log_metric('AUC', auc)

        # log atrifacts and model
        mlflow.log_artifact('plots/ROC_curve.png')

        # log model
        mlflow.sklearn.log_model(
            sk_model=model, 
            name=model_name, 
            serialization_format="skops",
            # Explicitly trust the internal scikit-learn metric/scorer types
            skops_trusted_types=[
                "sklearn.metrics._scorer._passthrough_scorer",
                "sklearn.metrics._classification.f1_score",
                "sklearn.metrics._classification.accuracy_score",
                "sklearn.metrics._ranking.roc_auc_score",
                "sklearn.metrics._scorer._Scorer"
                ]
            )

        mlflow.end_run()



test_features = label_encoder(X_test, categorical_cols)
test_target = y_test.map({'Y': 1, 'N': 0})

print('Logging Decision Tree')
mlflow_logging(model_dt, test_features, test_target, "DecisionTreeClassifier")

print('Logging Logistic Regression')
mlflow_logging(model_lr, test_features, test_target, "LogisticRegression")

print('Logging Random Forest')
mlflow_logging(model_rf, test_features, test_target, "RandomForestClassifier")

print('All models done..!!')

