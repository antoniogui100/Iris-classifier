from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model
model = joblib.load("iris_model.pkl")

# Create API app
app = FastAPI()

# Define input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Iris Classifier API is running!"}

@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = model.predict(features)[0]
    return {"prediction": int(prediction)}
