import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# FIX: Use relative import to correctly reference data_processing.py within the 'src' package.
from .data_processing import RFMAggregator 

# --- Configuration ---
PROCESSED_SCALED_DATA_PATH = os.path.join('data', 'processed', 'customer_features.csv')
RAW_DATA_PATH = os.path.join('data', 'raw', 'data.csv')
FINAL_MODEL_DATA_PATH = os.path.join('data', 'processed', 'model_ready_data.csv')

def interpret_clusters(df_unscaled, cluster_label, rfm_features):
    """Calculates the mean RFM values for each cluster to identify the 'worst' segment."""
    # Group by the cluster label and calculate the mean for interpretation
    cluster_means = df_unscaled.groupby(cluster_label)[rfm_features].mean().reset_index()
    
    print("\n--- Cluster Mean RFM Values (Unscaled) ---")
    print("Goal: Identify the cluster with high Recency (Worse), low Frequency (Worse), and low Monetary (Worse).")
    
    # Sort to place the worst cluster at the top
    # Recency (Days) should be sorted DESCENDING
    # Frequency and Total_Value should be sorted ASCENDING
    worst_cluster_row = cluster_means.sort_values(
        by=['Recency', 'Frequency', 'Total_Value'],
        ascending=[False, True, True]
    ).iloc[0]
    
    worst_cluster_id = worst_cluster_row[cluster_label]
    
    # Display the full sorted table for documentation
    print(cluster_means.sort_values(by=['Recency', 'Frequency', 'Total_Value'], ascending=[False, True, True]))
    
    return worst_cluster_id

def create_proxy_target():
    print("Starting Proxy Target Variable Engineering (Task 4)...")
    
    # 1. Load Scaled Features from Task 3
    try:
        X_scaled = pd.read_csv(PROCESSED_SCALED_DATA_PATH).set_index('CustomerId')
    except FileNotFoundError:
        print(f"Error: Scaled features not found at {PROCESSED_SCALED_DATA_PATH}. Please ensure Task 3 ran successfully.")
        return

    # 2. Identify the core RFM features used for clustering
    rfm_scaled_cols = ['Recency', 'Frequency', 'Total_Value', 'Avg_Value'] 
    X_rfm_scaled = X_scaled[[col for col in rfm_scaled_cols if col in X_scaled.columns]]

    # 3. K-Means Clustering (Instruction 2)
    K = 3 # Segment customers into 3 distinct groups [cite: 188]
    # Use a random_state for reproducibility [cite: 190]
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=300) 
    X_scaled['Cluster_Label'] = kmeans.fit_predict(X_rfm_scaled)
    
    print(f"Clustering complete. Found {K} clusters.")
    # 

    # 4. Interpret Clusters using Unscaled Data
    raw_df = pd.read_csv(RAW_DATA_PATH)
    
    # Instantiate and transform using the imported RFMAggregator to get unscaled metrics
    rfm_aggregator = RFMAggregator()
    X_rfm_unscaled = rfm_aggregator.fit_transform(raw_df) 
    
    # Merge cluster labels with unscaled RFM features
    rfm_unscaled_with_labels = X_rfm_unscaled[['Recency', 'Frequency', 'Total_Value']].merge(
        X_scaled['Cluster_Label'], 
        left_index=True, 
        right_index=True
    )
    
    # Identify the high-risk cluster based on business logic [cite: 192]
    worst_cluster_id = interpret_clusters(
        rfm_unscaled_with_labels, 
        'Cluster_Label', 
        ['Recency', 'Frequency', 'Total_Value']
    )
    
    print(f"\nIdentified 'Worst' Cluster (High Risk Proxy): Cluster {worst_cluster_id}")
    
    # 5. Define and Assign the "High-Risk" Label (Instruction 3)
    # Assign 1 to the worst cluster, 0 otherwise [cite: 193, 194]
    X_scaled['is_high_risk'] = (X_scaled['Cluster_Label'] == worst_cluster_id).astype(int)
    
    high_risk_count = X_scaled['is_high_risk'].sum()
    total_customers = len(X_scaled)
    print(f"\nTarget Variable created. High Risk Count: {high_risk_count} out of {total_customers}")
    print(f"High Risk Percentage: {X_scaled['is_high_risk'].mean() * 100:.2f}%")
    
    # 6. Final Data Preparation and Saving (Instruction 4)
    X_model_ready = X_scaled.drop(columns=['Cluster_Label']) 
    
    # Save the final model-ready dataset [cite: 196]
    X_model_ready.to_csv(FINAL_MODEL_DATA_PATH)
    print(f"\nFinal model-ready data saved to: {FINAL_MODEL_DATA_PATH}")

if __name__ == '__main__':
    # To run this script correctly with the relative import, you must execute it as a module
    print("--- IMPORTANT RUNNING INSTRUCTION ---")
    print("To avoid 'ImportError', please run this script as a module from the project root directory:")
    print("python -m src.proxy_target_engineering")
    print("-------------------------------------")
    
    try:
        create_proxy_target()
    except ImportError as e:
        print(f"\nFATAL ERROR: The script failed due to the import error: {e}")
        print("Please ensure you run the script using the 'python -m src.proxy_target_engineering' command.")