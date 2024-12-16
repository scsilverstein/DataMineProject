import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from .geo_visualization import create_geographic_visualization

def visualize_state_predictions(actual_df, predicted_df, save_path):
    """
    Create geographic visualizations comparing actual vs predicted values
    and prediction errors by state
    """
    # Ensure we're working with single values per state
    actual_df = actual_df.groupby('State')['Total'].mean().reset_index()
    predicted_df = predicted_df.groupby('State')['Total'].mean().reset_index()
    
    # Calculate prediction error
    error_df = pd.DataFrame({
        'State': actual_df['State'],
        'Error': predicted_df['Total'].values - actual_df['Total'].values,
        'Error_Pct': ((predicted_df['Total'].values - actual_df['Total'].values) 
                     / actual_df['Total'].values * 100),
        'Actual': actual_df['Total'].values,
        'Predicted': predicted_df['Total'].values
    })
    
    # Create visualizations
    # 1. Actual Values Map
    create_geographic_visualization(
        actual_df,
        'Total',
        'Actual Asthma Cases by State',
        f'{save_path}actual_map.png',
        cmap='YlOrBr'
    )
    
    # 2. Predicted Values Map
    create_geographic_visualization(
        predicted_df,
        'Total',
        'Predicted Asthma Cases by State',
        f'{save_path}predicted_map.png',
        cmap='YlOrBr'
    )
    
    # 3. Error Map
    create_geographic_visualization(
        error_df,
        'Error_Pct',
        'Prediction Error by State (%)',
        f'{save_path}error_map.png',
        cmap='RdYlBu_r'  # Red for overprediction, Blue for underprediction
    )
    
    # Calculate metrics
    metrics_df = pd.DataFrame({
        'State': actual_df['State'],
        'R2': r2_score(actual_df['Total'], predicted_df['Total']),
        'Mean_Error': error_df['Error'],
        'Mean_Error_Pct': error_df['Error_Pct']
    })
    
    # Save metrics
    metrics_df.to_csv(f'{save_path}state_metrics.csv', index=False)
    
    return metrics_df