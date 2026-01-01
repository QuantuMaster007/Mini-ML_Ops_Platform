from pydantic import BaseModel

class PredictRequest(BaseModel):
    age: float
    income: float

class PredictResponse(BaseModel):
    prediction: float
    latency_ms: float
    model_uri: str
