import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="Titanic Survival Predictor API", version="1.0.0")

# Load assets safely using absolute paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "titanic_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model_scaler.pkl")

# Standard production safety check
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model artifact missing at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

# Define input schema with validation boundaries
class PassengerFeatures(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Ticket class (1, 2, or 3)")
    Age: float = Field(..., ge=0.0, le=120.0, description="Age of the passenger")
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    Fare: float = Field(..., ge=0.0, description="Passenger fare")
    IsMale: int = Field(..., ge=0, le=1, description="1 if male, 0 if female")

@app.get("/health")
def health_check():
    """Liveness probe for AWS container orchestrators."""
    return {"status": "healthy"}

@app.post("/predict")
def predict_survival(passenger: PassengerFeatures):
    try:
        # Convert incoming JSON payload to raw array structure matching training
        raw_features = np.array([[
            passenger.Pclass,
            passenger.Age,
            passenger.SibSp,
            passenger.Parch,
            passenger.Fare,
            passenger.IsMale
        ]])
        
        # Apply transformation if data was scaled during training
        if scaler:
            processed_features = scaler.transform(raw_features)
        else:
            processed_features = raw_features

        # Execute inference
        prediction = int(model.predict(processed_features)[0])
        probability = float(model.predict_proba(processed_features)[0][1])

        return {
            "survived": prediction,
            "survival_probability": round(probability, 4)
        }
    except Exception as e:
        # Log the actual exception to CloudWatch internally, return generic 500
        raise HTTPException(status_code=500, detail="Internal inference engine failure.")