from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def perform_clustering(df, save_path='figures/'):
    """Perform clustering analysis with both linear and RF predictions"""
    # Prepare features
    features = ['Total', 'Total_Emissions']
    X = df[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Add predictions from both models
    linear_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    X_emissions = df[['Total_Emissions']].values
    y = df['Total'].values
    
    linear_model.fit(X_emissions, y)
    rf_model.fit(X_emissions, y)
    
    df['Linear_Pred'] = linear_model.predict(X_emissions)
    df['RF_Pred'] = rf_model.predict(X_emissions)
    
    # Linear Regression Clusters Plot
    plt.figure(figsize=(10, 6))
    scatter0 = plt.scatter(df['Total_Emissions'], df['Total'], 
                          c=df['Cluster'], cmap='viridis')
    plt.xlabel('Emissions')
    plt.ylabel('Asthma Cases')
    plt.title('Linear Regression: Clusters by Emissions and Cases')
    plt.colorbar(scatter0, label='Cluster')
    plt.savefig(f'{save_path}linear_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Random Forest Clusters Plot
    plt.figure(figsize=(10, 6))
    scatter1 = plt.scatter(df['Total_Emissions'], df['Total'], 
                          c=df['Cluster'], cmap='viridis')
    plt.xlabel('Emissions')
    plt.ylabel('Asthma Cases')
    plt.title('Random Forest: Clusters by Emissions and Cases')
    plt.colorbar(scatter1, label='Cluster')
    plt.savefig(f'{save_path}rf_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # State Distribution Heatmap
    plt.figure(figsize=(10, 6))
    cluster_counts = df.groupby(['State', 'Cluster']).size().unstack(fill_value=0)
    sns.heatmap(cluster_counts, cmap='YlOrRd', annot=True, fmt='d')
    plt.title('State Distribution in Clusters')
    plt.xticks(rotation=45)
    plt.savefig(f'{save_path}state_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Prediction Error Correlation
    plt.figure(figsize=(8, 6))
    error_df = pd.DataFrame({
        'Linear_Error': np.abs(df['Linear_Pred'] - df['Total']),
        'RF_Error': np.abs(df['RF_Pred'] - df['Total'])
    })
    sns.heatmap(error_df.corr(), annot=True, cmap='viridis')
    plt.title('Prediction Error Correlation')
    plt.savefig(f'{save_path}error_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Cluster Characteristics
    plt.figure(figsize=(8, 6))
    cluster_means = df.groupby('Cluster')[features].mean()
    sns.heatmap(cluster_means, annot=True, cmap='viridis')
    plt.title('Cluster Characteristics')
    plt.savefig(f'{save_path}cluster_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model Performance by Cluster
    plt.figure(figsize=(8, 6))
    cluster_performance = pd.DataFrame({
        'Linear_RMSE': [np.sqrt(mean_squared_error(
            df[df['Cluster'] == c]['Total'],
            df[df['Cluster'] == c]['Linear_Pred']
        )) for c in range(3)],
        'RF_RMSE': [np.sqrt(mean_squared_error(
            df[df['Cluster'] == c]['Total'],
            df[df['Cluster'] == c]['RF_Pred']
        )) for c in range(3)]
    }, index=[f'Cluster {i}' for i in range(3)])
    
    sns.heatmap(cluster_performance, annot=True, cmap='viridis')
    plt.title('Model Performance by Cluster (RMSE)')
    plt.savefig(f'{save_path}model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print cluster insights
    print("\nCluster Analysis Results:")
    for cluster in range(3):
        cluster_states = df[df['Cluster'] == cluster]['State'].unique()
        print(f"\nCluster {cluster}:")
        print(f"Number of states: {len(cluster_states)}")
        print(f"Average emissions: {df[df['Cluster'] == cluster]['Total_Emissions'].mean():.2f}")
        print(f"Average asthma cases: {df[df['Cluster'] == cluster]['Total'].mean():.2f}")
        print("States:", ', '.join(cluster_states))
        print(f"Linear RMSE: {cluster_performance.iloc[cluster]['Linear_RMSE']:.2f}")
        print(f"RF RMSE: {cluster_performance.iloc[cluster]['RF_RMSE']:.2f}")
    
    return df, cluster_means