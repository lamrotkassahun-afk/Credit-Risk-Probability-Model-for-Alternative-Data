import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# --- Configuration ---
FINAL_MODEL_DATA_PATH = os.path.join('data', 'processed', 'model_ready_data.csv')
RANDOM_STATE = 42

# --- Model Training and Evaluation ---

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluates the model and logs metrics to MLflow."""
    y_pred = model.predict(X_test)
    # Ensure y_proba generation works if the model supports it
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_proba = np.zeros_like(y_test, dtype=float) # Fallback if model doesn't support proba

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba.any() else 0.0
    }
    
    print(f"\n--- {model_name} Metrics ---")
    for key, value in metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")
        mlflow.log_metric(key, value)
        
    return metrics

def train_and_log_models():
    print("Starting Model Training and MLflow Tracking (Task 5)...")
    
    # 1. Data Preparation
    try:
        data = pd.read_csv(FINAL_MODEL_DATA_PATH).set_index('CustomerId')
    except FileNotFoundError:
        print(f"Error: Model ready data not found at {FINAL_MODEL_DATA_PATH}. Ensure Task 4 ran successfully.")
        return

    X = data.drop('is_high_risk', axis=1)
    y = data['is_high_risk']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Set up MLflow tracking
    mlflow.set_experiment("Credit_Risk_Scoring_Model")

    # --- Model 1: Logistic Regression (Interpretable Baseline) ---
    with mlflow.start_run(run_name="Logistic_Regression_GridSearch") as run:
        print("\nTraining Logistic Regression...")
        
        # Hyperparameter Tuning
        param_grid_lr = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs'],
            'penalty': ['l2']
        }
        grid_search_lr = GridSearchCV(
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 
            param_grid_lr, 
            cv=3, 
            scoring='roc_auc'
        )
        grid_search_lr.fit(X_train, y_train)
        best_lr = grid_search_lr.best_estimator_

        # Log parameters
        mlflow.log_params(best_lr.get_params())
        mlflow.log_param("Model_Type", "LogisticRegression")
        
        # Evaluate and Log metrics
        metrics_lr = evaluate_model(best_lr, X_test, y_test, "Logistic Regression")

        # Log the model
        mlflow.sklearn.log_model(best_lr, "logistic_regression_model")
        lr_model_uri = f"runs:/{run.info.run_id}/logistic_regression_model"
        mlflow.register_model(lr_model_uri, "CreditRiskModel")
        print(f"Logged Logistic Regression model in run: {run.info.run_id}")

    # --- Model 2: Gradient Boosting (High Performance) ---
    with mlflow.start_run(run_name="XGBoost_Default_Params") as run:
        print("\nTraining XGBoost Classifier...")
        
        # Train with default parameters (a simple version)
        xgb_model = XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False, 
            eval_metric='logloss', 
            random_state=RANDOM_STATE
        )
        xgb_model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_params(xgb_model.get_params())
        mlflow.log_param("Model_Type", "XGBoostClassifier")

        # Evaluate and Log metrics
        metrics_xgb = evaluate_model(xgb_model, X_test, y_test, "XGBoost Classifier")
        
        # Log the model
        mlflow.sklearn.log_model(xgb_model, "xgb_model")
        xgb_model_uri = f"runs:/{run.info.run_id}/xgb_model"
        mlflow.register_model(xgb_model_uri, "CreditRiskModel")
        print(f"Logged XGBoost model in run: {run.info.run_id}")
        
    # Model Comparison and Selection
    
    # Simple selection based on ROC-AUC for immediate use
    if metrics_lr['roc_auc'] > metrics_xgb['roc_auc']:
        best_model_name = "Logistic Regression"
    else:
        best_model_name = "XGBoost Classifier"
        
    print(f"\nModel Selection: {best_model_name} had the best ROC-AUC score.")
    
    # Instruct user to finalize registration via UI for production stage
    print("Next Step: Proceed to Task 6 (Deployment and CI/CD).")

if __name__ == '__main__':
    # Ensure MLflow backend is set up for persistent tracking
    # Default is local filesystem tracking in ./mlruns
    train_and_log_models()