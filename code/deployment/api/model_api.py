import os
import sys
from typing import Optional
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field

# Ensure local modules can be resolved
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models")))
sys.path.append(os.path.abspath("code/models"))
sys.path.append(os.path.abspath("."))

from feature_engineering import engineer_features


def find_path(rel_path: str) -> str:
    possible_paths = [
        rel_path,
        os.path.join("/app", rel_path),
        os.path.join("../../..", rel_path),
        os.path.join("../..", rel_path),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return rel_path


MODEL_PATH = os.getenv("MODEL_PATH", find_path("models/catboost_model.cbm"))

# Load global model instance
model = CatBoostClassifier()
if os.path.exists(MODEL_PATH):
    model.load_model(MODEL_PATH)
    print(f"Loaded CatBoost model from: {MODEL_PATH}")
else:
    print(f"Warning: Model file not found at '{MODEL_PATH}'")


app = FastAPI(
    title="Customer Churn Prediction API",
    description="FastAPI service for customer churn inference",
    version="1.0.0",
)


class CustomerInput(BaseModel):
    gender: str = Field(default="Female", description="Customer gender: Female or Male")
    SeniorCitizen: int = Field(default=0, description="Senior citizen flag (1, 0)")
    Partner: str = Field(default="No", description="Has partner: Yes or No")
    Dependents: str = Field(default="No", description="Has dependents: Yes or No")
    tenure: int = Field(default=1, description="Tenure in months")
    PhoneService: str = Field(default="Yes", description="Phone service: Yes or No")
    MultipleLines: str = Field(default="No", description="Multiple lines: No, Yes, or No phone service")
    InternetService: str = Field(default="DSL", description="Internet service: DSL, Fiber optic, No")
    OnlineSecurity: str = Field(default="No", description="Online security: Yes, No, No internet service")
    OnlineBackup: str = Field(default="No", description="Online backup: Yes, No, No internet service")
    DeviceProtection: str = Field(default="No", description="Device protection: Yes, No, No internet service")
    TechSupport: str = Field(default="No", description="Tech support: Yes, No, No internet service")
    StreamingTV: str = Field(default="No", description="Streaming TV: Yes, No, No internet service")
    StreamingMovies: str = Field(default="No", description="Streaming movies: Yes, No, No internet service")
    Contract: str = Field(default="Month-to-month", description="Contract: Month-to-month, One year, Two year")
    PaperlessBilling: str = Field(default="Yes", description="Paperless billing: Yes or No")
    PaymentMethod: str = Field(default="Electronic check", description="Payment method")
    MonthlyCharges: float = Field(default=29.85, description="Monthly charges amount")
    TotalCharges: Optional[float] = Field(default=None, description="Total charges (optional)")



@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "service": "Customer Churn Prediction API",
        "model_loaded": os.path.exists(MODEL_PATH),
    }


@app.post("/pipeline/reload")
def reload_model():
    """Hot-reloads the newly trained CatBoost model from disk without restarting container."""
    global model
    target_path = os.getenv("MODEL_PATH", find_path("models/catboost_model.cbm"))
    if os.path.exists(target_path):
        new_model = CatBoostClassifier()
        new_model.load_model(target_path)
        model = new_model
        print(f"Hot-reloaded model successfully from {target_path}")
        return {
            "status": "SUCCESS",
            "message": f"Model hot-reloaded successfully from {target_path}",
        }
    raise HTTPException(status_code=404, detail=f"Model artifact not found at {target_path}")


@app.post("/predict")
def predict(input_data: CustomerInput):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")

    data = input_data.model_dump()
    if data["TotalCharges"] is None:
        data["TotalCharges"] = float(data["MonthlyCharges"] * data["tenure"]) if data["tenure"] > 0 else 0.0

    df_raw = pd.DataFrame([data])
    df_features = engineer_features(df_raw)

    prediction = model.predict(df_features)
    probabilities = model.predict_proba(df_features)

    pred_class = int(prediction[0])
    churn_prob = float(probabilities[0][1])

    return {
        "prediction": pred_class,
        "churn": bool(pred_class == 1),
        "churn_probability": round(churn_prob, 4),
        "risk_level": "High" if churn_prob >= 0.5 else "Low",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)



