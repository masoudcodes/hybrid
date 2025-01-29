# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from typing import List, Optional
from enum import Enum

# Create FastAPI instance
app = FastAPI(title="Credit Risk Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
try:
    lr_model = joblib.load('logistic_regression_model.pkl')
    ann_model = tf.keras.models.load_model('credit_risk_ann_model.h5')
    print("Successfully loaded both models")
except Exception as e:
    print(f"Error loading models: {e}")
    raise Exception("Failed to load required model files. Please ensure both 'logistic_regression_model.pkl' and 'credit_risk_ann_model.h5' are present in the project directory.")

# Enums for categorical variables
class HomeOwnership(str, Enum):
    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"

class LoanIntent(str, Enum):
    PERSONAL = "PERSONAL"
    EDUCATION = "EDUCATION"
    MEDICAL = "MEDICAL"
    VENTURE = "VENTURE"
    HOME_IMPROVEMENT = "HOME_IMPROVEMENT"
    DEBT_CONSOLIDATION = "DEBT_CONSOLIDATION"

class LoanGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"

class DefaultHistory(str, Enum):
    Y = "Y"
    N = "N"

# Pydantic models for request/response
class CreditRequest(BaseModel):
    person_age: int = Field(..., ge=18, le=100, description="Age of the person")
    person_income: float = Field(..., ge=0, description="Annual income")
    person_home_ownership: HomeOwnership
    person_emp_length: float = Field(..., ge=0, description="Employment length in years")
    loan_intent: LoanIntent
    loan_grade: LoanGrade
    loan_amnt: float = Field(..., ge=0, description="Loan amount requested")
    loan_int_rate: float = Field(..., ge=0, le=100, description="Interest rate")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Loan amount as percentage of income")
    cb_person_default_on_file: DefaultHistory
    cb_person_cred_hist_length: float = Field(..., ge=0, description="Credit history length in years")

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    loan_status: str
    details: dict

# Hybrid model class
class HybridModel:
    def __init__(self, lr_model, ann_model, lr_weight=0.4, ann_weight=0.6):
        self.lr_model = lr_model
        self.ann_model = ann_model
        self.lr_weight = lr_weight
        self.ann_weight = ann_weight
    
    def predict_proba(self, X):
        lr_pred = self.lr_model.predict_proba(X)[:, 1]
        ann_pred = self.ann_model.predict(X).ravel()
        return self.lr_weight * lr_pred + self.ann_weight * ann_pred
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)

# Initialize hybrid model
hybrid_model = HybridModel(lr_model, ann_model)

# Preprocessing function
def preprocess_input(data: dict) -> dict:
    categorical_mappings = {
        'person_home_ownership': {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3},
        'loan_intent': {
            'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3,
            'HOME_IMPROVEMENT': 4, 'DEBT_CONSOLIDATION': 5
        },
        'loan_grade': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6},
        'cb_person_default_on_file': {'N': 0, 'Y': 1}
    }
    
    processed_data = data.copy()
    for field, mapping in categorical_mappings.items():
        processed_data[field] = mapping[processed_data[field]]
    
    return processed_data

# API Routes
@app.get("/")
async def root():
    return {
        "message": "Credit Risk Prediction API is running",
        "status": "active",
        "version": "1.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": bool(lr_model and ann_model)
    }

@app.get("/model-info")
async def get_model_info():
    return {
        "model_type": "Hybrid (Logistic Regression + Neural Network)",
        "features": {
            "person_age": "Age of the person (18-100)",
            "person_income": "Annual income",
            "person_home_ownership": list(HomeOwnership),
            "person_emp_length": "Employment length in years",
            "loan_intent": list(LoanIntent),
            "loan_grade": list(LoanGrade),
            "loan_amnt": "Loan amount requested",
            "loan_int_rate": "Interest rate",
            "loan_percent_income": "Loan amount as percentage of income (0-1)",
            "cb_person_default_on_file": list(DefaultHistory),
            "cb_person_cred_hist_length": "Credit history length in years"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_credit_risk(request: CreditRequest):
    try:
        # Convert request to dictionary and preprocess
        data_dict = request.dict()
        processed_data = preprocess_input(data_dict)
        
        # Convert to DataFrame
        input_df = pd.DataFrame([processed_data])
        
        # Make prediction
        probability = float(hybrid_model.predict_proba(input_df)[0])
        prediction = int(hybrid_model.predict(input_df)[0])
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low Risk"
        elif probability < 0.7:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
        
        # Create detailed response
        details = {
            "input_data": data_dict,
            "risk_probability": f"{probability:.2%}",
            "risk_factors": {
                "high_amount_to_income": data_dict["loan_percent_income"] > 0.4,
                "poor_credit_history": data_dict["cb_person_default_on_file"] == "Y",
                "short_employment": data_dict["person_emp_length"] < 2
            }
        }
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            loan_status="Default Risk" if prediction == 1 else "No Default Risk",
            details=details
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    