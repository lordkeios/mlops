import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from prefect import task
import os
from dotenv import load_dotenv
load_dotenv('.env')

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5050")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000")
DATASET_STORAGE_URL_PREFIX = os.getenv("DATASET_STORAGE_URL_PREFIX", "s3://mlops")

print(f"MLFLOW_S3_ENDPOINT_URL={MLFLOW_S3_ENDPOINT_URL}")
print(f"DATASET_STORAGE_URL_PREFIX={DATASET_STORAGE_URL_PREFIX}")

STORAGE_OPTIONS={
    "key": AWS_ACCESS_KEY_ID,
    "secret": AWS_SECRET_ACCESS_KEY,
    "client_kwargs": {"endpoint_url": MLFLOW_S3_ENDPOINT_URL}
}

# Set a unified random_state across the file
random_state = 100

@task(name="load_training_data")
def load_train_data(train_data_path):
    # Start by loading the train data
    try:
        data = pd.read_csv(train_data_path)
        print("Loading train data locally..")

    except:
        data = pd.read_csv(f'{DATASET_STORAGE_URL_PREFIX}/{train_data_path}', storage_options=STORAGE_OPTIONS)
        print("Loading train data from S3 Bucket..")

    # Randomly shuffle the data to minimise the effect of randomness on our results
    data = data.sample(frac=1.0, random_state=random_state)

    return data


def load_new_data(test_data_path):
    try:
        new_data = pd.read_csv(test_data_path)
        print("Loading test data locally..")
    except:
        new_data = pd.read_csv(f'{DATASET_STORAGE_URL_PREFIX}/{test_data_path}', storage_options=STORAGE_OPTIONS)
        print("Loading test data from S3 Bucket..")

    return new_data


def prepare_data(data):
    # Clean up column names
    data.columns = data.columns.str.strip()

    # Select features and target
    X = data[data.columns.difference(['class'])]
    Y = data['class']
    
    return X, Y


def feature_selector():
    # Select the top 10 most important features using chi-squared
    selector = SelectKBest(chi2, k=10)
    return selector


def create_data_preprocessor(all_feature_columns):
    # Select categorical and continuous columns
    numerical_columns = ['duration', 'credit_amount', 'age']
    categorical_columns = [c for c in all_feature_columns if c not in numerical_columns]

    # Define the steps in the categorical processor.
    categorical_processor = Pipeline(steps=[
        ('ordinal_encoder', OrdinalEncoder(dtype=np.int64, 
                                            handle_unknown='use_encoded_value', 
                                            unknown_value=-1)),
    ])

    # # Define the steps in the numerical processor (mainly for Logistic Regression)
    # numerical_processor = Pipeline(steps=[
    #         ('numerical_selector', FeatureSelector(numerical)),
    #         # ('Standardscaler', StandardScaler()),
    # ])

    # Combine the categorical and numerical processors into a combined processor
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical_processor', categorical_processor, categorical_columns),
            # ('numerical_processor', numerical_processor, numerical_columns),
            ('passthrough', 'passthrough', numerical_columns),
            ],
        remainder='drop', verbose_feature_names_out=False)
    
    return preprocessor
