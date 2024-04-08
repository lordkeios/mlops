from sklearn.model_selection import cross_val_predict
import mlflow
from prefect import task

from .data_preparation import prepare_data, load_new_data

import os
from dotenv import load_dotenv
load_dotenv('.env')

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5050")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000")
DATASET_STORAGE_URL_PREFIX = os.getenv("DATASET_STORAGE_URL_PREFIX", "s3://mlops")

STORAGE_OPTIONS={
    "key": AWS_ACCESS_KEY_ID,
    "secret": AWS_SECRET_ACCESS_KEY,
    "client_kwargs": {"endpoint_url": MLFLOW_S3_ENDPOINT_URL}
}


def make_cv_predictions(model_pipeline, X, Y):
    # Make cross-validated predictions 
    predictions = cross_val_predict(model_pipeline, X, Y, cv=10, n_jobs=-1, verbose=3)
    return predictions


def save_model_predictions(predictions, predictions_path):
    # Save model predictions
    try:
        predictions.to_csv(predictions_path, index=False)
        print("Saving predictions locally..")
    except:
        predictions.to_csv(f'{DATASET_STORAGE_URL_PREFIX}/{predictions_path}', storage_options=STORAGE_OPTIONS, index=False)
        print("Saving predictions to S3 Bucket..")


def make_predictions(model_pipeline, data, predictions_path):
    # Prepare the data as before
    X, Y = prepare_data(data)

    # Make and store model predictions
    data['predictions'] = model_pipeline.predict(X)
    data['prediction_probs'] = model_pipeline.predict_proba(X)[:, 1]
    save_model_predictions(data, predictions_path)

    return data


@task(name="make_predictions_with_model_registry_model")
def make_predictions_with_model_registry_model(model_name, data_path, output_path, 
                                               stage="Production"):
    """ Function for retreiving model from the Model Registry 
        and make predictions with it. 
    """

    # Load the data and prepare it
    data = load_new_data(data_path)
    X, Y = prepare_data(data)

    # Load the model and make predictions
    try:
        model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{stage}")
        data['predictions'] = model.predict(X)
        data['prediction_probs'] = model.predict_proba(X)[:, 1]
        
        save_model_predictions(data, output_path)
    except Exception as e:
        print("\n***\nNo predictions were made; No model in Production stage.\n***\n")
        print(e)

    return data
