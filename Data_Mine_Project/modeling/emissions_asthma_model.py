import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization_utils import add_model_labels, set_style

class EmissionsAsthmaModel:
    def __init__(self):
        self.linear_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare the datasets"""
        # Load datasets
        asthma_df = pd.read_csv('../CSV Files/AsthmaTotals.csv')
        emissions_df = pd.read_csv('../CSV Files/emissions.csv')
        
        # Process emissions data
        emissions_total = emissions_df[
            (emissions_df['sector-name'] == 'Total carbon dioxide emissions from all sectors') &
            (emissions_df['fuel-name'] == 'All Fuels')
        ][['year', 'state-name', 'value']].rename(
            columns={'year': 'Year', 'state-name': 'State', 'value': 'Total_Emissions'}
        )
        
        # Merge datasets
        merged_df = pd.merge(asthma_df, emissions_total, on=['State', 'Year'])
        
        return merged_df
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        X = df[['Total_Emissions']].values
        y = df['Total'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_models(self, X_train, y_train):
        """Train both Linear Regression and Random Forest models"""
        # Train Linear Regression
        self.linear_model.fit(X_train, y_train)
        
        # Train Random Forest
        self.rf_model.fit(X_train, y_train)
    
    def visualize_results(self, df, X_test, y_test, save_path='figures/'):
        """Create visualizations comparing both models"""
        # Get predictions from both models
        y_pred_linear = self.linear_model.predict(X_test)
        y_pred_rf = self.rf_model.predict(X_test)
        
        # 1. Actual vs Predicted Comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Linear Regression plot
        ax1.scatter(y_test, y_pred_linear, alpha=0.5)
        ax1.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', label='Perfect Prediction')
        ax1.set_xlabel('Actual Asthma Cases')
        ax1.set_ylabel('Predicted Asthma Cases')
        ax1.set_title('Linear Regression: Actual vs Predicted')
        ax1.legend()
        
        # Random Forest plot
        ax2.scatter(y_test, y_pred_rf, alpha=0.5)
        ax2.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', label='Perfect Prediction')
        ax2.set_xlabel('Actual Asthma Cases')
        ax2.set_ylabel('Predicted Asthma Cases')
        ax2.set_title('Random Forest: Actual vs Predicted')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_path}model_predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Residuals Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Linear Regression residuals
        residuals_linear = y_test - y_pred_linear
        ax1.scatter(y_pred_linear, residuals_linear, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Linear Regression: Residuals Plot')
        
        # Random Forest residuals
        residuals_rf = y_test - y_pred_rf
        ax2.scatter(y_pred_rf, residuals_rf, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Random Forest: Residuals Plot')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}model_residuals_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Error Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Linear Regression error distribution
        sns.histplot(residuals_linear, ax=ax1, bins=30)
        ax1.set_title('Linear Regression: Error Distribution')
        ax1.set_xlabel('Prediction Error')
        
        # Random Forest error distribution
        sns.histplot(residuals_rf, ax=ax2, bins=30)
        ax2.set_title('Random Forest: Error Distribution')
        ax2.set_xlabel('Prediction Error')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}model_errors_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print model performance metrics
        print("\nModel Performance Metrics:")
        
        print("\nLinear Regression:")
        print(f"R² Score: {r2_score(y_test, y_pred_linear):.3f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_linear)):.3f}")
        
        print("\nRandom Forest:")
        print(f"R² Score: {r2_score(y_test, y_pred_rf):.3f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.3f}")
        
        # 4. Feature Importance Comparison (for Random Forest)
        plt.figure(figsize=(8, 6))
        feature_importance = pd.DataFrame({
            'Feature': ['Total_Emissions'],
            'Importance': self.rf_model.feature_importances_
        })
        sns.barplot(data=feature_importance, x='Feature', y='Importance')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{save_path}rf_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close() 