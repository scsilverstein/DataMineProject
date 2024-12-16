import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedHealthModeling:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_and_merge_data(self):
        """Load and merge all relevant datasets"""
        # Load datasets
        asthma_df = pd.read_csv('./CSV Files/AsthmaTotals.csv')
        emissions_df = pd.read_csv('./CSV Files/emissions.csv')
        disease_df = pd.read_csv('./CSV Files/Disease Data .csv')
        mortality_df = pd.read_csv('./CSV Files/LEAB.csv')
        
        # Process emissions
        emissions = emissions_df[
            emissions_df['sector-name'] == 'Total carbon dioxide emissions from all sectors'
        ][['year', 'state-name', 'fuel-name', 'value']].pivot_table(
            index=['year', 'state-name'], 
            columns='fuel-name', 
            values='value'
        ).reset_index()
        
        # Process disease data
        disease_pivot = disease_df.pivot_table(
            index='Year',
            columns='Disease Name',
            values='Number of Diagnoses'
        ).reset_index()
        
        # Merge datasets
        merged_df = pd.merge(
            asthma_df,
            emissions,
            left_on=['Year', 'State'],
            right_on=['year', 'state-name']
        )
        
        return merged_df
    
    def create_time_series_model(self, df):
        """Time series forecasting for health outcomes"""
        from statsmodels.tsa.arima.model import ARIMA
        
        # Group by year and calculate mean rates
        yearly_rates = df.groupby('Year')['Asthma_Rate'].mean()
        
        # Fit ARIMA model
        model = ARIMA(yearly_rates, order=(1,1,1))
        results = model.fit()
        
        # Make predictions
        forecast = results.forecast(steps=5)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_rates.index, yearly_rates.values, label='Historical')
        plt.plot(
            range(yearly_rates.index[-1], yearly_rates.index[-1] + 6),
            forecast,
            '--', label='Forecast'
        )
        plt.title('Time Series Forecast of Asthma Rates')
        plt.xlabel('Year')
        plt.ylabel('Asthma Rate')
        plt.legend()
        plt.show()
        
        return results
    
    def create_multi_target_model(self, df):
        """Predict multiple health outcomes simultaneously"""
        from sklearn.multioutput import MultiOutputRegressor
        
        # Prepare multiple targets
        targets = ['Asthma_Rate', 'Mortality_Rate']
        features = ['Coal', 'Natural Gas', 'Petroleum']
        
        X = df[features]
        y = df[targets]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Create and train model
        model = MultiOutputRegressor(RandomForestRegressor())
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for i, target in enumerate(targets):
            axes[i].scatter(y_test.iloc[:, i], y_pred[:, i])
            axes[i].plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
                        [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
                        'r--')
            axes[i].set_title(f'Predicted vs Actual {target}')
        plt.tight_layout()
        plt.show()
        
        return model
    
    def create_neural_network(self, df):
        """Deep learning approach for health prediction"""
        # Prepare features
        features = ['Coal', 'Natural Gas', 'Petroleum', 'Year']
        X = df[features]
        y = df['Asthma_Rate']
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
        
        # Create and train neural network
        nn_model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=1000
        )
        nn_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = nn_model.predict(X_test)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                'r--')
        plt.title('Neural Network Predictions')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.show()
        
        return nn_model
    
    def create_geospatial_model(self, df):
        """Incorporate geographic features into prediction"""
        
        # Create pipeline with preprocessing
        numeric_features = ['Coal', 'Natural Gas', 'Petroleum', 'Year']
        categorical_features = ['State']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first'), categorical_features)
            ])
        
        # Create and train model
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor())
        ])
        
        # Split and fit
        X = df[numeric_features + categorical_features]
        y = df['Asthma_Rate']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Plot regional analysis
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='Region', y='Asthma_Rate')
        plt.title('Asthma Rates by Region')
        plt.xticks(rotation=45)
        plt.show()
        
        return model

def run_advanced_analysis():
    """Run all advanced modeling approaches"""
    analyzer = AdvancedHealthModeling()
    df = analyzer.load_and_merge_data()
    
    # Time series analysis
    print("Running time series analysis...")
    time_series_model = analyzer.create_time_series_model(df)
    
    # Multi-target prediction
    print("\nRunning multi-target prediction...")
    multi_target_model = analyzer.create_multi_target_model(df)
    
    # Neural network
    print("\nTraining neural network...")
    nn_model = analyzer.create_neural_network(df)
    
    # Geospatial analysis
    print("\nRunning geospatial analysis...")
    geo_model = analyzer.create_geospatial_model(df)
    
    return {
        'time_series': time_series_model,
        'multi_target': multi_target_model,
        'neural_network': nn_model,
        'geospatial': geo_model
    }

if __name__ == "__main__":
    models = run_advanced_analysis() 