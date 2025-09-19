from fastapi import FastAPI
from pydantic import BaseModel
import pickle

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Simple 4-feature Classifier")

class InputFeatures(BaseModel):
    f1: float
    f2: float
    f3: float
    f4: float

@app.post("/predict")
def predict(input_data: InputFeatures):
    data = [[
        input_data.f1,
        input_data.f2,
        input_data.f3,
        input_data.f4
    ]]
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
