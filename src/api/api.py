import os
import uvicorn
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

# --- Configuration ---
MLFLOW_MODEL_NAME = "CreditRiskModel"

# The 22 feature names used by the model, matching the columns in model_ready_data.csv 
# These must match the features trained in Task 5 exactly (order and name).
FEATURE_NAMES = [
    'Recency', 'Frequency', 'Total_Value', 'Avg_Value', 'Std_Amount', 
    'Total_Amount', 'numerical__Recency', 'numerical__Frequency', 
    'numerical__Total_Value', 'numerical__Avg_Value', 'numerical__Std_Amount', 
    'numerical__Total_Amount', 'categorical__Most_Used_Channel_Channel_1', 
    'categorical__Most_Used_Channel_Channel_2', 'categorical__Most_Used_Channel_Channel_3', 
    'categorical__Most_Used_Category_Category 1', 'categorical__Most_Used_Category_Category 2', 
    'categorical__Most_Used_Category_Missing', 'categorical__Most_Used_Pricing_1', 
    'categorical__Most_Used_Pricing_2', 'categorical__Most_Used_Pricing_3', 
    'categorical__Most_Used_Pricing_4'
]

# --- Pydantic Schema for Input Data ---
# A simpler schema using a dictionary that contains all 22 feature values.
class CustomerFeatures(BaseModel):
    features: Dict[str, float]

# --- API Setup ---
app = FastAPI(title="Credit Risk Scoring API", version="1.0")
model = None

# --- Model Loading ---
def load_model():
    """Load the latest registered model from MLflow Model Registry."""
    global model
    try:
        # Load the latest model (Version 2, the XGBoost model, should be used)
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Successfully loaded model from MLflow: {model_uri}")
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        # In a real environment, this would raise a critical error
        raise

@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts."""
    load_model()
    
# --- Prediction Endpoint (Instruction 2) ---
@app.post("/predict_risk/")
def predict_risk(customer_data: CustomerFeatures):
    """
    Predicts the credit risk (1 for High Risk, 0 for Low Risk) 
    and returns the probability score.
    """
    if model is None:
        return {"error": "Model not loaded. Check MLflow server status."}

    try:
        # 1. Convert input dictionary to a DataFrame, ensuring correct column order
        feature_values = [customer_data.features.get(name, 0.0) for name in FEATURE_NAMES]
        df_input = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
        
        # 2. Predict probability (risk score)
        # XGBoost output: [Prob_Low_Risk, Prob_High_Risk]
        risk_proba = model.predict_proba(df_input)[:, 1][0]
        
        # 3. Predict class (0 or 1)
        risk_class = int(model.predict(df_input)[0])

        return {
            "model_used": MLFLOW_MODEL_NAME,
            "predicted_class": risk_class,
            "high_risk_probability": round(risk_proba, 4),
            "risk_status": "High Risk" if risk_class == 1 else "Low Risk"
        }

    except Exception as e:
        return {"error": f"Prediction failed: {e}"}


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure MLflow tracking server is running (mlflow ui) before running this
    # Uvicorn will start the FastAPI server
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)