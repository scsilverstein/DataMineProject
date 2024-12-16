from emissions_asthma_model import EmissionsAsthmaModel
from cluster_analysis import perform_clustering
from modeling_analysis import EmissionsHealthAnalysis
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os

def ensure_figures_directory():
    """Create figures directory if it doesn't exist"""
    if not os.path.exists('figures'):
        os.makedirs('figures')

def run_complete_analysis():
    """Run complete modeling pipeline with both Linear and RF models"""
    print("Starting analysis pipeline...")
    ensure_figures_directory()
    
    try:
        # 1. Linear and RF Analysis
        print("\n1. Running emissions-asthma analysis...")
        model = EmissionsAsthmaModel()
        df = model.load_data()
        X, y = model.prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train both models
        model.train_models(X_train, y_train)
        
        # Visualize results
        model.visualize_results(df, X_test, y_test, save_path='figures/')
        
        # 2. Clustering Analysis
        print("\n2. Performing clustering analysis...")
        cluster_df, cluster_means = perform_clustering(df)
        
        # 3. Statistical Analysis
        print("\n3. Running statistical analysis...")
        analyzer = EmissionsHealthAnalysis()
        asthma_corr, state_correlations = analyzer.analyze_correlations(df)
        analyzer.visualize_results({'correlations': (asthma_corr, state_correlations)})
        
        # 4. State-level Predictions
        print("\n4. Running state-level predictions...")
        predictions, metrics = analyzer.predict_by_state(df)
        
        # 5. Future Predictions
        print("\n5. Generating future predictions...")
        future_predictions = analyzer.predict_future(
            df, 
        )
        
        # 6. Model Comparison
        print("\n6. Comparing model performance...")
        model_comparison = analyzer.compare_models(df)
        
        # 7. Future Predictions Comparison
        print("\n7. Comparing future predictions...")
        future_comparison = analyzer.predict_future_comparison(
            df,
            save_path='figures/future_'
        )
        
        # Save results
        results = {
            'regression_metrics': {
                'linear': {
                    'r2': model.linear_model.score(X_test, y_test),
                    'coefficients': model.linear_model.coef_,
                    'intercept': model.linear_model.intercept_
                },
                'random_forest': {
                    'r2': model.rf_model.score(X_test, y_test)
                }
            },
            'clustering': {
                'cluster_means': cluster_means.to_dict(),
                'state_clusters': cluster_df[['State', 'Cluster']].to_dict()
            },
            'correlations': {
                'asthma': asthma_corr,
                'state_correlations': state_correlations
            },
            'predictions': predictions,
            'future_predictions': future_predictions,
            'model_comparison': model_comparison,
            'future_comparison': future_comparison
        }
        
        pd.to_pickle(results, 'analysis_results.pkl')
        print("\nAnalysis complete. Results saved to 'analysis_results.pkl'")
        print("Visualizations saved in 'figures' directory")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    run_complete_analysis()