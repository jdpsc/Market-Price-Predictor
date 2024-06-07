import os

import mlflow


def make_predictions(model, data):

    predictions = model.predict(data)

    return predictions


def load_model_from_registry(model_name: str):

    # Load a model from the mlflow model registry
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@champion")

    return model
