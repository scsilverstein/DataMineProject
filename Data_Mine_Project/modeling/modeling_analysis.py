import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from utils.geo_visualization import create_geographic_visualization, create_multi_map_visualization, create_model_comparison_map
from utils.prediction_visualization import visualize_state_predictions
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import geopandas as gpd
from utils.visualization_utils import set_style
states = gpd.read_file('./cb_2018_us_state_20m.zip')
class EmissionsHealthAnalysis:
    def __init__(self):
        self.linear_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def analyze_correlations(self, df, save_path='figures/'):
        """Analyze correlations between emissions and health outcomes"""
        # Asthma correlations with total emissions
        asthma_corr = {
            'Total_Emissions': df['Total_Emissions'].corr(df['Total'])
        }
        
        # State-level analysis
        state_correlations = []
        for state in df['State'].unique():
            state_data = df[df['State'] == state]
            if len(state_data) > 2:  # Need at least 3 points for correlation
                corr = state_data['Total_Emissions'].corr(state_data['Total'])
                state_correlations.append({
                    'State': state,
                    'Correlation': corr
                })
        
        # Print correlation insights
        print("\nCorrelation Analysis Results:")
        print(f"Correlation between Total Emissions and Asthma Cases: {asthma_corr['Total_Emissions']:.3f}")
        
        return asthma_corr, state_correlations
    
    def visualize_results(self, results, save_path='figures/'):
        """Create visualizations for both models"""
        asthma_corr, state_correlations = results['correlations']
        
        # Convert state correlations list to dataframe
        comparison_df = pd.DataFrame(state_correlations)
        
        # Create geographic visualization for single correlation map
        create_geographic_visualization(
            comparison_df,
            'Correlation',
            'Emissions-Asthma Correlation by State',
            f'{save_path}correlation_map.png',
            cmap='RdYlBu'
        )
        
        # Print summary statistics
        print("\nCorrelation Analysis Summary:")
        print(f"Overall Emissions-Asthma Correlation: {asthma_corr['Total_Emissions']:.3f}")
        
        # Top positive correlations
        print("\nTop 5 States by Correlation Strength:")
        top_states = comparison_df.nlargest(5, 'Correlation')
        for _, row in top_states.iterrows():
            print(f"{row['State']}: {row['Correlation']:.3f}")
            
        # Top negative correlations
        print("\nTop 5 States with Negative Correlations:")
        bottom_states = comparison_df.nsmallest(5, 'Correlation')
        for _, row in bottom_states.iterrows():
            print(f"{row['State']}: {row['Correlation']:.3f}")
        
        # Create correlation distribution plot
        plt.figure(figsize=(12, 6))
        sns.histplot(data=comparison_df, x='Correlation', bins=20)
        plt.title('Distribution of State-level Correlations')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Count')
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.savefig(f'{save_path}correlation_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return comparison_df
    
    def predict_by_state(self, df, save_path='figures/'):
        """Perform and visualize state-level predictions"""
        # First, get the most recent year's data for each state
        latest_year = df['Year'].max()
        latest_data = df[df['Year'] == latest_year].copy()
        
        # Group historical data by state for training
        predictions = {}
        state_metrics = []
        
        for state in df['State'].unique():
            state_data = df[df['State'] == state].copy()
            
            if len(state_data) > 2:  # Need at least 3 points for training
                # Prepare features
                X = state_data[['Total_Emissions']].values
                y = state_data['Total'].values
                
                # Train model on all historical data
                model = LinearRegression()
                model.fit(X, y)
                
                # Make prediction for latest year
                latest_X = latest_data[latest_data['State'] == state][['Total_Emissions']].values
                if len(latest_X) > 0:
                    latest_pred = model.predict(latest_X)[0]
                    
                    predictions[state] = {
                        'actual': state_data['Total'].iloc[-1],
                        'predicted': latest_pred,
                        'r2': r2_score(y, model.predict(X)),
                        'coefficient': model.coef_[0],
                        'intercept': model.intercept_
                    }
        
        # Create dataframes for visualization using only the latest year's data
        actual_df = pd.DataFrame({
            'State': latest_data['State'],
            'Total': latest_data['Total']
        })
        
        predicted_df = pd.DataFrame({
            'State': latest_data['State'],
            'Total': [predictions[state]['predicted'] 
                     if state in predictions else np.nan 
                     for state in latest_data['State']]
        })
        
        # Visualize predictions
        metrics_df = visualize_state_predictions(
            actual_df,
            predicted_df,
            save_path
        )
        
        # Print summary
        print("\nState-level Prediction Results:")
        print(f"\nPredictions for {latest_year}:")
        print("\nTop 5 States by Prediction Accuracy (R²):")
        print(metrics_df.nlargest(5, 'R2')[['State', 'R2', 'Mean_Error_Pct']])
        
        print("\nStates with Largest Prediction Errors (%):")
        print(metrics_df.nlargest(5, 'Mean_Error_Pct')[['State', 'Mean_Error_Pct']])
        
        return predictions, metrics_df
    
    def predict_future(self, df, target_years=[2020, 2021,2022,2024,2025, 2030, 2040, 2100], save_path='figures/'):
        """Predict future asthma cases for specified years"""
        print("\nGenerating Future Predictions...")
        
        # Get the latest actual data point for each state
        latest_year = df['Year'].max()
        latest_data = df[df['Year'] == latest_year].copy()
        
        # Calculate average yearly change in emissions for each state
        emissions_trends = {}
        for state in df['State'].unique():
            state_data = df[df['State'] == state].sort_values('Year')
            if len(state_data) > 1:
                yearly_change = np.polyfit(state_data['Year'], 
                                         state_data['Total_Emissions'], 
                                         deg=1)[0]
                emissions_trends[state] = yearly_change
        
        future_predictions = {}
        for year in target_years:
            predictions = {}
            years_ahead = year - latest_year
            
            for state in df['State'].unique():
                state_data = df[df['State'] == state].copy()
                
                if len(state_data) > 2:  # Need at least 3 points for training
                    # Train model on historical data
                    X = state_data[['Total_Emissions']].values
                    y = state_data['Total'].values
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Project emissions based on trend
                    if state in emissions_trends:
                        projected_emissions = (latest_data[latest_data['State'] == state]['Total_Emissions'].iloc[0] + 
                                            emissions_trends[state] * years_ahead)
                        
                        # Make prediction
                        future_pred = model.predict([[projected_emissions]])[0]
                        
                        predictions[state] = {
                            'projected_emissions': projected_emissions,
                            'predicted_cases': future_pred,
                            'emissions_trend': emissions_trends[state]
                        }
            
            future_predictions[year] = predictions
        
        # Create visualizations for each future year
        for year, predictions in future_predictions.items():
            pred_df = pd.DataFrame({
                'State': list(predictions.keys()),
                'Total': [p['predicted_cases'] for p in predictions.values()],
                'Emissions': [p['projected_emissions'] for p in predictions.values()]
            })
            
            # Create map of predicted cases
            create_geographic_visualization(
                pred_df,
                'Total',
                f'Predicted Asthma Cases by State ({year})',
                f'{save_path}prediction_{year}_map.png',
                cmap='YlOrBr'
            )
        
        # Print summary statistics
        print("\nFuture Predictions Summary:")
        for year in target_years:
            total_cases = sum(p['predicted_cases'] 
                            for p in future_predictions[year].values())
            print(f"\nYear {year}:")
            print(f"Projected Total US Asthma Cases: {total_cases:,.0f}")
            
            # Top 5 states by predicted cases
            state_predictions = [(state, pred['predicted_cases']) 
                               for state, pred in future_predictions[year].items()]
            top_5 = sorted(state_predictions, key=lambda x: x[1], reverse=True)[:5]
            
            print("\nTop 5 States by Predicted Cases:")
            for state, cases in top_5:
                print(f"{state}: {cases:,.0f}")
        
        # Create trend visualization
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        historical_total = df.groupby('Year')['Total'].sum()
        plt.plot(historical_total.index, historical_total.values, 
                'b-', label='Historical Data')
        
        # Plot future predictions
        future_years = list(future_predictions.keys())
        future_totals = [sum(p['predicted_cases'] 
                           for p in future_predictions[year].values())
                        for year in future_years]
        
        plt.plot(future_years, future_totals, 'r--', label='Predictions')
        plt.scatter(future_years, future_totals, c='red')
        
        plt.title('US Total Asthma Cases: Historical Data and Future Predictions')
        plt.xlabel('Year')
        plt.ylabel('Total Asthma Cases')
        plt.ylim(0, 5000000)
        plt.grid(True)
        plt.legend()
        set_style()
        
        # Add value annotations
        for i, year in enumerate(future_years):
            plt.annotate(f'{future_totals[i]:,.0f}', 
                        (year, future_totals[i]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}future_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return future_predictions
    
    def compare_models(self, df, save_path='figures/'):
        """Compare Linear Regression and Random Forest models"""
        print("\nComparing Linear Regression and Random Forest Models...")
        
        # Prepare data
        latest_year = df['Year'].max()
        model_results = {'Linear': {}, 'Random Forest': {}}
        
        for state in df['State'].unique():
            state_data = df[df['State'] == state].copy()
            
            if len(state_data) > 2:
                X = state_data[['Total_Emissions']].values
                y = state_data['Total'].values
                
                # Train both models
                self.linear_model.fit(X, y)
                self.rf_model.fit(X, y)
                
                # Make predictions
                linear_pred = self.linear_model.predict(X)
                rf_pred = self.rf_model.predict(X)
                
                # Store results
                model_results['Linear'][state] = {
                    'r2': r2_score(y, linear_pred),
                    'rmse': np.sqrt(mean_squared_error(y, linear_pred)),
                    'predictions': linear_pred
                }
                
                model_results['Random Forest'][state] = {
                    'r2': r2_score(y, rf_pred),
                    'rmse': np.sqrt(mean_squared_error(y, rf_pred)),
                    'predictions': rf_pred
                }
        
        # Create comparison visualizations
        # 1. R² Score Comparison
        plt.figure(figsize=(15, 6))
        comparison_data = []
        for model_name in ['Linear', 'Random Forest']:
            for state, results in model_results[model_name].items():
                comparison_data.append({
                    'Model': model_name,
                    'State': state,
                    'R² Score': results['r2']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        sns.boxplot(data=comparison_df, x='Model', y='R² Score')
        plt.title('Model Performance Comparison (R² Score)')
        plt.savefig(f'{save_path}model_r2_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. State-by-State Comparison
        plt.figure(figsize=(15, 8))
        for model_name in ['Linear', 'Random Forest']:
            r2_scores = [results['r2'] 
                        for results in model_results[model_name].values()]
            plt.hist(r2_scores, alpha=0.5, label=model_name, bins=20)
        plt.title('Distribution of R² Scores by Model')
        plt.xlabel('R² Score')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(f'{save_path}model_r2_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print("\nModel Performance Summary:")
        for model_name in ['Linear', 'Random Forest']:
            r2_scores = [results['r2'] 
                        for results in model_results[model_name].values()]
            rmse_scores = [results['rmse'] 
                          for results in model_results[model_name].values()]
            
            print(f"\n{model_name} Model:")
            print(f"Average R² Score: {np.mean(r2_scores):.3f}")
            print(f"Average RMSE: {np.mean(rmse_scores):.3f}")
            
            # Best performing states
            state_scores = [(state, results['r2']) 
                           for state, results in model_results[model_name].items()]
            top_states = sorted(state_scores, key=lambda x: x[1], reverse=True)[:5]
            
            print("\nTop 5 States by R² Score:")
            for state, score in top_states:
                print(f"{state}: {score:.3f}")
        
        return model_results
    
    def predict_future_comparison(self, df, save_path='figures/'):
        """Compare Linear and Random Forest predictions including historical years"""
        print("\nGenerating Complete Prediction Comparison...")
        
       
        latest_year = df['Year'].max()
        
        # Create detailed near-term predictions (2014-2025)
             # Long-term predictions
        all_years = list(range(2014, 2025)) + [2025, 2030, 2040, 2100] 
        
        predictions = {
            'Linear': {year: {} for year in all_years},
            'Random Forest': {year: {} for year in all_years}
        }
        
        # Train models and make predictions for each state
        for state in df['State'].unique():
            state_data = df[df['State'] == state].copy()
            
            if len(state_data) > 2:
                X = state_data[['Total_Emissions']].values
                y = state_data['Total'].values
                
                # Train both models on all available data
                self.linear_model.fit(X, y)
                self.rf_model.fit(X, y)
                
                # Calculate emissions trend
                emissions_trend = np.polyfit(state_data['Year'], 
                                           state_data['Total_Emissions'], 
                                           deg=1)[0]
                
                # Make predictions for all years
                for year in all_years:
                    if year <= latest_year:
                        # Use actual emissions data if available
                        year_data = state_data[state_data['Year'] == year]
                        if not year_data.empty:
                            emissions = year_data['Total_Emissions'].values[0]
                        else:
                            # Interpolate if year is missing
                            years_diff = year - state_data['Year'].min()
                            emissions = (state_data['Total_Emissions'].iloc[0] + 
                                      emissions_trend * years_diff)
                    else:
                        # Project future emissions
                        years_ahead = year - latest_year
                        emissions = (state_data['Total_Emissions'].iloc[-1] + 
                                   emissions_trend * years_ahead)
                    
                    # Make predictions with both models
                    predictions['Linear'][year][state] = self.linear_model.predict([[emissions]])[0]
                    predictions['Random Forest'][year][state] = self.rf_model.predict([[emissions]])[0]
        
        # Create prediction maps for all years
        for year in all_years:
            self.create_prediction_maps(predictions, year, save_path)
        
        # Visualize complete prediction timeline
        plt.figure(figsize=(20, 10))
        
        # Plot actual historical data
        historical_data = df.groupby('Year')['Total'].sum()
        plt.plot(historical_data.index, historical_data.values, 
                'k-', label='Actual Data', linewidth=2)
        
        # Plot predictions for each model
        styles = {
            'Linear': {'color': 'b', 'linestyle': '--', 'marker': 'o'},
            'Random Forest': {'color': 'r', 'linestyle': '--', 'marker': 's'}
        }
        
        for model_name in ['Linear', 'Random Forest']:
            # Calculate totals for all years
            predicted_totals = [sum(predictions[model_name][year].values()) 
                              for year in all_years]
            
          
            # Plot near-term predictions

            # Plot long-term predictions
            plt.plot(all_years, predicted_totals,
                    color=styles[model_name]['color'],
                    linestyle=styles[model_name]['linestyle'],
                    label=f"{model_name} (2014-2100)")
            
            # Add markers and annotations for all predictions
            for years, predictions_array in [(all_years, predicted_totals)]:
                plt.scatter(years, predictions_array,
                          color=styles[model_name]['color'],
                          marker=styles[model_name]['marker'])
                
                # Add annotations for key years
                for year, total in zip(years, predictions_array):
                    if year in [2016, 2020, 2025, 2030, 2040, 2100]:
                        plt.annotate(f'{total:,.0f}',
                                   (year, total),
                                   textcoords="offset points",
                                   xytext=(0,10),
                                   ha='center',
                                   color=styles[model_name]['color'])
        
        plt.title('US Total Asthma Cases: Historical Data and Predictions (2014-2100)')
        plt.xlabel('Year')
        plt.ylabel('Total Asthma Cases')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format y-axis with comma separator
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Add vertical lines for key years
        plt.axvline(x=latest_year, color='gray', linestyle=':', alpha=0.5, label='Latest Data')
        plt.axvline(x=2025, color='gray', linestyle='--', alpha=0.5, label='Near-term/Long-term Split')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}complete_predictions_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print detailed comparison summary
        print("\nPrediction Summary:")
        
        # Near-term predictions (2014-2025)
       
        # Long-term predictions
        print("\nLong-term Predictions:")
        for year in all_years:
            print(f"\nYear {year}:")
            linear_total = sum(predictions['Linear'][year].values())
            rf_total = sum(predictions['Random Forest'][year].values())
            print(f"Linear Model: {linear_total:,.0f}")
            print(f"Random Forest: {rf_total:,.0f}")
            print(f"Difference: {abs(linear_total - rf_total):,.0f} "
                  f"({abs(linear_total - rf_total)/linear_total*100:.1f}%)")
        
        return predictions
    
    def analyze_feature_importance(self, df, save_path='figures/'):
        """Analyze feature importance using both models"""
        print("\nAnalyzing Feature Importance...")
        
        # Prepare features
        features = ['Total_Emissions', 'Year']
        X = df[features]
        y = df['Total']
        
        # Train both models
        self.rf_model.fit(X, y)
        self.linear_model.fit(X, y)
        
        # Random Forest Feature Importance
        rf_importance = pd.DataFrame({
            'Feature': features,
            'Importance': self.rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Linear Regression Coefficients (normalized)
        coef_importance = np.abs(self.linear_model.coef_)
        linear_importance = pd.DataFrame({
            'Feature': features,
            'Importance': coef_importance / np.sum(coef_importance)
        }).sort_values('Importance', ascending=False)
        
        # Visualize feature importance comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(121)
        sns.barplot(data=rf_importance, x='Feature', y='Importance')
        plt.title('Random Forest Feature Importance')
        plt.xticks(rotation=45)
        
        plt.subplot(122)
        sns.barplot(data=linear_importance, x='Feature', y='Importance')
        plt.title('Linear Regression Coefficient Importance')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print feature importance insights
        print("\nFeature Importance Rankings:")
        print("\nRandom Forest:")
        for _, row in rf_importance.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.3f}")
        
        print("\nLinear Regression:")
        for _, row in linear_importance.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.3f}")
        
        return rf_importance, linear_importance
    
    def analyze_regional_patterns(self, df, save_path='figures/'):
        """Analyze regional variations using both models"""
        print("\nAnalyzing Regional Patterns...")
        
        # Define regions
        regions = {
            'Northeast': ['Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 'Rhode Island', 
                        'Connecticut', 'New York', 'Pennsylvania', 'New Jersey'],
            'Midwest': ['Wisconsin', 'Michigan', 'Illinois', 'Indiana', 'Ohio', 'Missouri', 
                       'North Dakota', 'South Dakota', 'Nebraska', 'Kansas', 'Minnesota', 'Iowa'],
            'South': ['Delaware', 'Maryland', 'Virginia', 'West Virginia', 'Kentucky', 
                     'North Carolina', 'South Carolina', 'Tennessee', 'Georgia', 'Florida', 
                     'Alabama', 'Mississippi', 'Arkansas', 'Louisiana', 'Oklahoma', 'Texas'],
            'West': ['Idaho', 'Montana', 'Wyoming', 'Nevada', 'Utah', 'Colorado', 'Arizona', 
                    'New Mexico', 'Alaska', 'Washington', 'Oregon', 'California', 'Hawaii']
        }
        
        # Add region column
        df['Region'] = df['State'].map({state: region 
                                      for region, states in regions.items() 
                                      for state in states})
        
        # Analyze each region
        regional_results = []
        for region in regions.keys():
            region_data = df[df['Region'] == region]
            X = region_data[['Total_Emissions']]
            y = region_data['Total']
            
            # Train both models
            self.rf_model.fit(X, y)
            self.linear_model.fit(X, y)
            
            # Get predictions
            rf_pred = self.rf_model.predict(X)
            linear_pred = self.linear_model.predict(X)
            
            regional_results.append({
                'Region': region,
                'RF_R2': r2_score(y, rf_pred),
                'Linear_R2': r2_score(y, linear_pred),
                'RF_RMSE': np.sqrt(mean_squared_error(y, rf_pred)),
                'Linear_RMSE': np.sqrt(mean_squared_error(y, linear_pred))
            })
        
        regional_df = pd.DataFrame(regional_results)
        
        # Visualize regional performance
        plt.figure(figsize=(15, 6))
        
        plt.subplot(121)
        sns.barplot(data=regional_df.melt(id_vars=['Region'], 
                                        value_vars=['RF_R2', 'Linear_R2'],
                                        var_name='Model', value_name='R²'),
                   x='Region', y='R²', hue='Model')
        plt.title('Model Performance by Region (R²)')
        plt.xticks(rotation=45)
        
        plt.subplot(122)
        sns.barplot(data=regional_df.melt(id_vars=['Region'], 
                                        value_vars=['RF_RMSE', 'Linear_RMSE'],
                                        var_name='Model', value_name='RMSE'),
                   x='Region', y='RMSE', hue='Model')
        plt.title('Model Performance by Region (RMSE)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}regional_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print regional insights
        print("\nRegional Performance Comparison:")
        for _, row in regional_df.iterrows():
            print(f"\n{row['Region']}:")
            print(f"Random Forest - R²: {row['RF_R2']:.3f}, RMSE: {row['RF_RMSE']:.3f}")
            print(f"Linear Regression - R²: {row['Linear_R2']:.3f}, RMSE: {row['Linear_RMSE']:.3f}")
        
        return regional_df
    
    def create_prediction_maps(self, predictions, year, save_path='figures/'):
        """Create side-by-side maps comparing Linear and RF predictions for a given year"""
        # Prepare data for both models
        linear_data = pd.DataFrame({
            'State': list(predictions['Linear'][year].keys()),
            'Total': list(predictions['Linear'][year].values())
        })
        
        rf_data = pd.DataFrame({
            'State': list(predictions['Random Forest'][year].keys()),
            'Total': list(predictions['Random Forest'][year].values())
        })
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Load US states shapefile
        states['NAME'] = states['NAME'].apply(lambda x: x.title())
        
        # Merge data with geographic information
        linear_states = states.merge(linear_data, how='left', left_on='NAME', right_on='State')
        rf_states = states.merge(rf_data, how='left', left_on='NAME', right_on='State')
        
        # Filter for continental US
        continental_linear = linear_states[~linear_states['NAME'].isin(['Alaska', 'Hawaii'])]
        continental_rf = rf_states[~rf_states['NAME'].isin(['Alaska', 'Hawaii'])]
        
        # Plot Linear Regression predictions
        continental_linear.plot(column='Total',
                              ax=ax1,
                              legend=True,
                              legend_kwds={'label': 'Predicted Cases'},
                              missing_kwds={'color': 'lightgrey'},
                              cmap='YlOrRd')
        ax1.set_title(f'Linear Regression Predictions ({year})')
        ax1.axis('off')
        
        # Plot Random Forest predictions
        continental_rf.plot(column='Total',
                          ax=ax2,
                          legend=True,
                          legend_kwds={'label': 'Predicted Cases'},
                          missing_kwds={'color': 'lightgrey'},
                          cmap='YlOrRd')
        ax2.set_title(f'Random Forest Predictions ({year})')
        ax2.axis('off')
        
        # Add state labels to both maps
        for ax, data in [(ax1, continental_linear), (ax2, continental_rf)]:
            for idx, row in data.iterrows():
                centroid = row.geometry.centroid
                ax.annotate(text=row['STUSPS'], 
                           xy=(centroid.x, centroid.y),
                           horizontalalignment='center',
                           fontsize=8)
        
        # Set map bounds for both plots
        for ax in [ax1, ax2]:
            ax.set_xlim([-125, -65])
            ax.set_ylim([25, 50])
        
        plt.suptitle(f'Predicted Asthma Cases by State ({year})', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{save_path}prediction_comparison_{year}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create difference map
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        # Calculate prediction differences
        diff_data = pd.DataFrame({
            'State': linear_data['State'],
            'Difference': (rf_data['Total'].values - linear_data['Total'].values) / linear_data['Total'].values * 100
        })
        
        # Merge difference data with states
        diff_states = states.merge(diff_data, how='left', left_on='NAME', right_on='State')
        continental_diff = diff_states[~diff_states['NAME'].isin(['Alaska', 'Hawaii'])]
        
        # Plot differences
        continental_diff.plot(column='Difference',
                            ax=ax,
                            legend=True,
                            legend_kwds={'label': 'Difference (%)'},
                            missing_kwds={'color': 'lightgrey'},
                            cmap='RdYlBu')
        
        # Add state labels
        for idx, row in continental_diff.iterrows():
            centroid = row.geometry.centroid
            ax.annotate(text=row['STUSPS'], 
                       xy=(centroid.x, centroid.y),
                       horizontalalignment='center',
                       fontsize=8)
        
        ax.set_xlim([-125, -65])
        ax.set_ylim([25, 50])
        ax.axis('off')
        
        plt.title(f'Difference Between RF and Linear Predictions ({year})', pad=20)
        plt.savefig(f'{save_path}prediction_difference_{year}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()