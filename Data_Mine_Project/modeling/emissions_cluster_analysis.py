import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from DataMineProject.Data_Mine_Project.modeling.visualization_utils import add_model_labels, set_style

class EmissionsClusterAnalysis:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.dbscan = None
        
    def load_data(self):
        """Load and prepare the datasets"""
        # Load datasets
        asthma_df = pd.read_csv('./CSV Files/AsthmaTotals.csv')
        emissions_df = pd.read_csv('./CSV Files/emissions.csv')
        
        # Process emissions data
        emissions_total = emissions_df[
            (emissions_df['sector-name'] == 'Total carbon dioxide emissions from all sectors') &
            (emissions_df['fuel-name'] == 'All Fuels')
        ][['year', 'state-name', 'value']].rename(
            columns={'year': 'Year', 'state-name': 'State', 'value': 'Total_Emissions'}
        )
        
        # Merge datasets
        merged_df = pd.merge(asthma_df, emissions_total, on=['State', 'Year'], how='inner')
        merged_df['Asthma_Rate'] = merged_df['Total']  # or per capita if population available
        
        return merged_df
    
    def perform_clustering(self, df):
        """Perform multiple clustering analyses"""
        # Prepare features for clustering
        features = df[['Total_Emissions', 'Asthma_Rate']].values
        features_scaled = self.scaler.fit_transform(features)
        
        # K-means clustering
        # Find optimal number of clusters
        silhouette_scores = []
        K = range(2, 8)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features_scaled)
            score = silhouette_score(features_scaled, kmeans.labels_)
            silhouette_scores.append(score)
        
        optimal_k = K[np.argmax(silhouette_scores)]
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans_labels = self.kmeans.fit_predict(features_scaled)
        
        # DBSCAN clustering
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = self.dbscan.fit_predict(features_scaled)
        
        return {
            'features_scaled': features_scaled,
            'kmeans_labels': kmeans_labels,
            'dbscan_labels': dbscan_labels,
            'optimal_k': optimal_k,
            'silhouette_scores': silhouette_scores
        }
    
    def analyze_clusters(self, df, cluster_results):
        """Analyze characteristics of each cluster"""
        df['KMeans_Cluster'] = cluster_results['kmeans_labels']
        df['DBSCAN_Cluster'] = cluster_results['dbscan_labels']
        
        # Cluster statistics
        kmeans_stats = df.groupby('KMeans_Cluster').agg({
            'State': 'count',
            'Total_Emissions': ['mean', 'std'],
            'Asthma_Rate': ['mean', 'std']
        }).round(2)
        
        dbscan_stats = df.groupby('DBSCAN_Cluster').agg({
            'State': 'count',
            'Total_Emissions': ['mean', 'std'],
            'Asthma_Rate': ['mean', 'std']
        }).round(2)
        
        # State cluster assignments
        state_clusters = df.groupby('State')['KMeans_Cluster'].agg(lambda x: x.value_counts().index[0])
        
        return {
            'kmeans_stats': kmeans_stats,
            'dbscan_stats': dbscan_stats,
            'state_clusters': state_clusters
        }
    
    def visualize_clusters(self, df, cluster_results):
        """Create visualizations of clustering results"""
        set_style()
        
        # 1. K-means vs DBSCAN comparison with state labels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # K-means plot with state labels
        scatter1 = ax1.scatter(df['Total_Emissions'], df['Asthma_Rate'], 
                             c=cluster_results['kmeans_labels'], cmap='viridis',
                             s=100)  # Increased point size
        
        # Add state labels to K-means plot
        for idx, row in df.iterrows():
            ax1.annotate(row['State'], 
                        (row['Total_Emissions'], row['Asthma_Rate']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
            
        ax1.set_xlabel('Total Emissions', fontsize=12)
        ax1.set_ylabel('Asthma Rate', fontsize=12)
        ax1.set_title(f'K-means Clustering (k={cluster_results["optimal_k"]})', 
                     fontsize=14, pad=20)
        plt.colorbar(scatter1, ax=ax1, label='Cluster')
        
        # DBSCAN plot with state labels
        scatter2 = ax2.scatter(df['Total_Emissions'], df['Asthma_Rate'], 
                             c=cluster_results['dbscan_labels'], cmap='viridis',
                             s=100)  # Increased point size
        
        # Add state labels to DBSCAN plot
        for idx, row in df.iterrows():
            ax2.annotate(row['State'], 
                        (row['Total_Emissions'], row['Asthma_Rate']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
            
        ax2.set_xlabel('Total Emissions', fontsize=12)
        ax2.set_ylabel('Asthma Rate', fontsize=12)
        ax2.set_title('DBSCAN Clustering', fontsize=14, pad=20)
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
        
        add_model_labels(fig, "Clustering", "Emissions-Asthma Patterns")
        plt.tight_layout()
        plt.show()
        
        # 2. Silhouette score analysis
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, 8), cluster_results['silhouette_scores'], marker='o')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Optimal Number of Clusters Analysis')
        plt.tight_layout()
        plt.show()
        
        # 3. Geographic distribution of clusters
        plt.figure(figsize=(15, 6))
        state_clusters = df.groupby('State')['KMeans_Cluster'].agg(lambda x: x.value_counts().index[0])
        state_clusters = state_clusters.sort_values()
        
        sns.barplot(x=state_clusters.index, y=[1]*len(state_clusters), 
                   hue=state_clusters.values, dodge=False)
        plt.xticks(rotation=90)
        plt.ylabel('')
        plt.title('Geographic Distribution of Clusters')
        plt.tight_layout()
        plt.show()

def run_cluster_analysis():
    """Run the complete clustering analysis"""
    analyzer = EmissionsClusterAnalysis()
    
    # Load data
    df = analyzer.load_data()
    
    # Perform clustering
    cluster_results = analyzer.perform_clustering(df)
    
    # Analyze clusters
    analysis_results = analyzer.analyze_clusters(df, cluster_results)
    
    # Visualize results
    analyzer.visualize_clusters(df, cluster_results)
    
    # Print cluster insights
    print("\nCluster Analysis Results:")
    print("\nK-means Cluster Statistics:")
    print(analysis_results['kmeans_stats'])
    
    print("\nDBSCAN Cluster Statistics:")
    print(analysis_results['dbscan_stats'])
    
    print("\nState Cluster Assignments:")
    print(analysis_results['state_clusters'])

if __name__ == "__main__":
    run_cluster_analysis() 