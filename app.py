from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import requests

# Load trained model
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

# Load clustering model
CLUSTER_MODEL_PATH = "cluster_model.pkl"
cluster_model = joblib.load(CLUSTER_MODEL_PATH)

# Model download URL
MODEL_URL = "https://github.com/srikuntala9-droid/ml-fastapi/releases/download/v1.0-model/model.pkl"

if not os.path.exists(MODEL_PATH):
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

# Load trained model
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="AQI Classification API",
    description="FastAPI for AQI Category Prediction",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aqi-classification-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AQIInput(BaseModel):
    hour_00: float
    hour_01: float
    hour_02: float
    hour_03: float
    hour_04: float
    hour_05: float
    hour_06: float
    hour_07: float
    hour_08: float
    hour_09: float
    hour_10: float
    hour_11: float
    hour_12: float
    hour_13: float
    hour_14: float
    hour_15: float
    hour_16: float
    hour_17: float
    hour_18: float
    hour_19: float
    hour_20: float
    hour_21: float
    hour_22: float
    hour_23: float


@app.get("/")
def home():
    return {"message": "AQI Classification API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/cluster")
def cluster(data: AQIInput):

    features = np.array([[
        data.hour_00,
        data.hour_01,
        data.hour_02,
        data.hour_03,
        data.hour_04,
        data.hour_05,
        data.hour_06,
        data.hour_07,
        data.hour_08,
        data.hour_09,
        data.hour_10,
        data.hour_11,
        data.hour_12,
        data.hour_13,
        data.hour_14,
        data.hour_15,
        data.hour_16,
        data.hour_17,
        data.hour_18,
        data.hour_19,
        data.hour_20,
        data.hour_21,
        data.hour_22,
        data.hour_23
    ]])

    cluster_prediction = cluster_model.predict(features)[0]

    return {
        "cluster": int(cluster_prediction)
    }


    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    return {
        "prediction": prediction,
        "probabilities": probabilities.tolist()
    }
