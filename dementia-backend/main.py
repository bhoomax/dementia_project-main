from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
ann_model = joblib.load("models/ann2_model.pkl")
knn_model = joblib.load("models/knn_model.pkl")
logistic_model = joblib.load("models/logistic_model.pkl")
nbc_model = joblib.load("models/NBC_model.pkl")
adaboost_model = joblib.load("models/adaboost_model.pkl")
rfc_model = joblib.load("models/rfc2_model.pkl")
dectree_model = joblib.load("models/dectree_model.pkl")

# Load RFE feature mask (boolean array or indices)
rfe_mask = joblib.load("models/rfe_mask.pkl")

# Load scaler
scaler = joblib.load("models/scaler2.pkl")

# Input schema
class PatientData(BaseModel):
    gender: str
    age: float
    EDUC: float
    SES: float
    MMSE: float
    CDR: float
    eTIV: float
    nWBV: float
    ASF: float
    model_name: str

# Mapping result
result_mapping = {0: "Non-Demented", 1: "Demented"}

@app.post("/predict/")
def predict(data: PatientData):
    # Create DataFrame
    input_data = pd.DataFrame([[
        0 if data.gender.upper() == "M" else 1,
        data.age,
        data.EDUC,
        data.SES,
        data.MMSE,
        data.CDR,
        data.eTIV,
        data.nWBV,
        data.ASF
    ]], columns=['M/F','Age','EDUC','SES','MMSE','CDR','eTIV','nWBV','ASF'])

    # Scale all features
    input_scaled = scaler.transform(input_data)

    # Select RFE features only
    input_rfe = input_scaled[:, rfe_mask]

    # Run predictions
    results = {
        "ANN_Prediction": result_mapping[int((ann_model.predict(input_scaled) > 0.5).astype("int32").item())],
        "KNN_Prediction": result_mapping[int(knn_model.predict(input_rfe).item())],
        "Logistic_Regression_Prediction": result_mapping[int(logistic_model.predict(input_rfe).item())],
        "Naive_Bayes_Prediction": result_mapping[int(nbc_model.predict(input_rfe).item())],
        "AdaBoost_Prediction": result_mapping[int(adaboost_model.predict(input_rfe).item())],
        "Random_Forest_Prediction": result_mapping[int(rfc_model.predict(input_scaled).item())],  # All features
        "Decision_Tree_Prediction": result_mapping[int(dectree_model.predict(input_scaled).item())]  # All features
    }

    return results
