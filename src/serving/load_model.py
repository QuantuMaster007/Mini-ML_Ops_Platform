import os
import mlflow.pyfunc

def get_model_uri():
    name = os.getenv("MODEL_NAME", "mini-mlops-platform-model")
    stage = os.getenv("MODEL_STAGE", "production").lower()
    return f"models:/{name}@{stage}"

def load_model():
    uri = get_model_uri()
    model = mlflow.pyfunc.load_model(uri)
    return model, uri
