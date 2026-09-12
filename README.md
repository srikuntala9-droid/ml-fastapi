# AQI Classification API

## Project Overview
This project deploys a Machine Learning classification model using FastAPI.

The API predicts the Air Quality Index (AQI) category based on 24 hourly AQI values.

## Machine Learning Model
- Algorithm: Random Forest Classifier
- Input Features: 24 hourly AQI values
- Output: Predicted AQI category
- Model File: model.pkl

## API
Framework: FastAPI

### Endpoint

POST `/predict`

### Input
The API accepts 24 hourly AQI values from `hour_00` to `hour_23`.

### Output
The API returns:
- Prediction
- Class probabilities

## Project Files

- `app.py` - FastAPI application
- `model.pkl` - Trained Machine Learning model
- `requirements.txt` - Required Python packages
- `README.md` - Project documentation

## How to Run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 9001

