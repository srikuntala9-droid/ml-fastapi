from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="AQI Clustering API")

# Allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load clustering model
cluster_model = joblib.load("cluster_model.pkl")


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
    return {"message": "AQI Clustering API is running"}


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

    prediction = cluster_model.predict(features)

    return {
        "cluster": int(prediction[0])
    }
