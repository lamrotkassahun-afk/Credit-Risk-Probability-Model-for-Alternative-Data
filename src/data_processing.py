import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Configuration ---
RAW_DATA_PATH = os.path.join('data', 'raw', 'data.csv')
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'customer_features.csv')

# FIX: Define SNAPSHOT_DATE as timezone-aware (UTC) to match the TransactionStartTime data
# This resolves the "Cannot subtract tz-naive and tz-aware datetime-like objects" error.
SNAPSHOT_DATE = pd.to_datetime('2018-12-20').tz_localize('UTC')

# --- Custom Transformer for Task 3 Instructions 1, 2, 4 ---
class RFMAggregator(BaseEstimator, TransformerMixin):
    """
    A custom transformer to perform transaction-to-customer level aggregation (RFM) 
    and extract time-based features, while handling column dropping and time conversion.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 1. Drop redundant/constant columns identified in EDA
        X = X.drop(columns=['CountryCode', 'TransactionId', 'BatchId', 
                            'SubscriptionId', 'AccountId', 'CurrencyCode'], 
                   errors='ignore')

        # 2. Time Conversion and Feature Extraction
        # Convert to datetime; since the data contains 'Z', it's timezone-aware (UTC)
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        
        # Extract features (Note: Year will be constant)
        X['Hour'] = X['TransactionStartTime'].dt.hour
        X['Day'] = X['TransactionStartTime'].dt.day
        X['Month'] = X['TransactionStartTime'].dt.month
        
        # 3. Aggregate Features (Instruction 1)
        
        # Recency (R): Days since last transaction
        recency_df = X.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
        # The subtraction now works because both SNAPSHOT_DATE and the column are TZ-aware
        recency_df['Recency'] = (SNAPSHOT_DATE - recency_df['TransactionStartTime']).dt.days
        recency_df = recency_df[['CustomerId', 'Recency']]
        
        # Frequency (F) and Monetary (M) Aggregates
        agg_features = X.groupby('CustomerId').agg(
            # Frequency (F): Number of transactions
            Frequency=('TransactionStartTime', 'count'),
            # Monetary (M): Sum of absolute transaction value
            Total_Value=('Value', 'sum'),
            Avg_Value=('Value', 'mean'),
            # Standard deviation features for variability analysis (Risk)
            Std_Value=('Value', 'std'),
            Std_Amount=('Amount', 'std'),
            Total_Amount=('Amount', 'sum'),
            # Categorical Mode Features (for features we want to retain)
            Most_Used_Channel=('ChannelId', lambda x: x.mode()[0] if not x.mode().empty else 'Missing'),
            Most_Used_Category=('ProductCategory', lambda x: x.mode()[0] if not x.mode().empty else 'Missing'),
            Most_Used_Pricing=('PricingStrategy', lambda x: x.mode()[0] if not x.mode().empty else 'Missing')
        ).reset_index()
        
        # Merge R, F, M features
        customer_df = pd.merge(agg_features, recency_df, on='CustomerId', how='left')
        
        # Set all NaN standard deviations (customers with 1 transaction) to 0
        customer_df['Std_Amount'] = customer_df['Std_Amount'].fillna(0)
        
        # Drop the redundant Std_Value column
        customer_df = customer_df.drop(columns=['Std_Value'], errors='ignore')

        return customer_df.set_index('CustomerId')

# --- Main Pipeline Setup ---

def make_processing_pipeline(df):
    """
    Creates and executes the full feature engineering pipeline.
    """
    # 1. Apply the RFM Aggregation to get customer-level data
    customer_features = RFMAggregator().fit_transform(df)

    # Separate feature types for ColumnTransformer
    numerical_cols = ['Recency', 'Frequency', 'Total_Value', 'Avg_Value', 
                      'Std_Amount', 'Total_Amount']
    categorical_cols = ['Most_Used_Channel', 'Most_Used_Category', 'Most_Used_Pricing']
    
    # Filter for columns that actually exist in the aggregated DataFrame
    numerical_cols = [col for col in numerical_cols if col in customer_features.columns]
    categorical_cols = [col for col in categorical_cols if col in customer_features.columns]


    # --- Pipeline Steps (Instructions 3, 5) ---
    
    # Custom Log Transformer for highly skewed features (Instruction 5)
    class LogTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            X_out = X.copy()
            # Apply log(1+x) to highly skewed/monetary/frequency features
            for col in ['Total_Value', 'Avg_Value', 'Frequency']:
                if col in X_out.columns:
                    # Using np.log1p (log(1+x)) to handle potential zeros gracefully
                    X_out[col] = np.log1p(X_out[col])
            # Recency is left untransformed for simpler interpretation as days
            return X_out

    numerical_pipeline = Pipeline([
        ('log_transform', LogTransformer()),
        # Standardization (Instruction 5)
        ('scaler', StandardScaler())
    ])

    # Define categorical transformation (One-Hot Encoding) (Instruction 3)
    categorical_pipeline = Pipeline([
        # WoE/IV is usually done manually first; for pipeline, we start with OHE
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_pipeline, numerical_cols),
            ('categorical', categorical_pipeline, categorical_cols)
        ],
        remainder='drop' 
    )

    # --- Execution and Output ---
    
    # Fit and transform the data
    X_processed_array = preprocessor.fit_transform(customer_features)
    
    # Get feature names after One-Hot Encoding
    numerical_output_cols = numerical_cols
    ohe_cols = list(preprocessor.named_transformers_['categorical']['onehot'].get_feature_names_out(categorical_cols))
    feature_names = numerical_output_cols + ohe_cols
    
    # Convert array back to DataFrame with CustomerId as index
    X_processed_df = pd.DataFrame(X_processed_array, 
                                  columns=feature_names, 
                                  index=customer_features.index)
    
    print(f"Original transaction rows: {len(df)}")
    print(f"Processed customer features (rows: {len(X_processed_df)}, cols: {len(X_processed_df.columns)})")
    
    # Save the processed features for Task 4
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    X_processed_df.to_csv(PROCESSED_DATA_PATH)
    print(f"Processed features saved to: {PROCESSED_DATA_PATH}")
    
    # Return features and original customers data for next tasks
    return X_processed_df, customer_features

if __name__ == '__main__':
    print("Starting Feature Engineering (Task 3)...")
    
    # Load the raw data
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}.")
        print("Please ensure 'data.csv' is placed in the 'data/raw/' directory.")
        exit()
        
    # Execute the processing pipeline
    X_features, X_rfm_unscaled = make_processing_pipeline(raw_df)
    
    # Display sample output
    # FIX: Correctly terminated string literal (resolved previous SyntaxError)
    print("\n--- Sample of Processed Features (Scaled and One-Hot Encoded) ---") 
    print(X_features.head())
    print("\nNext Step: Proceed to Task 4 (Proxy Target Variable Engineering) using the features in data/processed/customer_features.csv")