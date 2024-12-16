import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

states = gpd.read_file('./cb_2018_us_state_20m.zip')
def create_geographic_visualization(data_df, column, title, save_path, cmap='viridis', model_name=None):
    """Create geographic visualization of state-level data"""
    # Add model name to title if provided
    if model_name:
        title = f"{model_name}: {title}"
    
    # Load US states shapefile
    
    # Convert state names to match our data
    states['NAME'] = states['NAME'].apply(lambda x: x.title())
    
    # Merge data with geographic information
    states_data = states.merge(data_df, how='left', left_on='NAME', right_on='State')
    
    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # Filter for continental US
    continental_us = states_data[~states_data['NAME'].isin(['Alaska', 'Hawaii'])]
    
    # Plot continental states
    continental_us.plot(column=column,
                       ax=ax,
                       legend=True,
                       legend_kwds={'label': column},
                       missing_kwds={'color': 'lightgrey'},
                       cmap=cmap)
    
    # Set map bounds for continental US
    ax.set_xlim([-125, -65])  # Longitude bounds
    ax.set_ylim([25, 50])     # Latitude bounds
    ax.axis('off')
    
    # Add title
    plt.title(title, fontsize=24, pad=20)
    
    # Add state labels
    for idx, row in continental_us.iterrows():
        centroid = row.geometry.centroid
        plt.annotate(text=row['STUSPS'], 
                    xy=(centroid.x, centroid.y),
                    horizontalalignment='center',
                    fontsize=14,
                    weight='bold')
    
    # Create inset maps for Alaska and Hawaii
    # Alaska inset
    ax_ak = fig.add_axes([0.01, 0.1, 0.2, 0.2])
    alaska = states_data[states_data['NAME'] == 'Alaska']
    if not alaska.empty:
        alaska.plot(column=column,
                   ax=ax_ak,
                   cmap=cmap)
        ax_ak.axis('off')
        ax_ak.set_title('Alaska', fontsize=16)
        
        # Add Alaska label
        ak_centroid = alaska.geometry.iloc[0].centroid
        ax_ak.annotate('AK', xy=(ak_centroid.x, ak_centroid.y),
                      horizontalalignment='center', fontsize=8)
    
    # Hawaii inset
    ax_hi = fig.add_axes([0.01, 0.35, 0.2, 0.2])
    hawaii = states_data[states_data['NAME'] == 'Hawaii']
    if not hawaii.empty:
        hawaii.plot(column=column,
                   ax=ax_hi,
                   cmap=cmap)
        ax_hi.axis('off')
        ax_hi.set_title('Hawaii', fontsize=16)
        
        # Add Hawaii label
        hi_centroid = hawaii.geometry.iloc[0].centroid
        ax_hi.annotate('HI', xy=(hi_centroid.x, hi_centroid.y),
                      horizontalalignment='center', fontsize=8)
    
    # Increase legend text size
    legend = ax.get_legend()
    if legend:
        legend.set_title(column, prop={'size': 20})
        for t in legend.get_texts():
            t.set_fontsize(16)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_multi_map_visualization(data_df, columns, titles, save_path, cmaps=None):
    """Create multiple geographic visualizations side by side"""
    if cmaps is None:
        cmaps = ['viridis'] * len(columns)
    
    # Load US states shapefile
    states['NAME'] = states['NAME'].apply(lambda x: x.title())
    
    # Merge data with geographic information
    states_data = states.merge(data_df, how='left', left_on='NAME', right_on='State')
    
    # Create subplots
    fig, axes = plt.subplots(1, len(columns), figsize=(20, 10))
    if len(columns) == 1:
        axes = [axes]
    
    # Filter for continental US
    continental_us = states_data[~states_data['NAME'].isin(['Alaska', 'Hawaii'])]
    
    # Create each map
    for ax, column, title, cmap in zip(axes, columns, titles, cmaps):
        continental_us.plot(column=column,
                          ax=ax,
                          legend=True,
                          legend_kwds={'label': column},
                          missing_kwds={'color': 'lightgrey'},
                          cmap=cmap)
        
        # Set map bounds for continental US
        ax.set_xlim([-125, -65])  # Longitude bounds
        ax.set_ylim([25, 50])     # Latitude bounds
        ax.axis('off')
        ax.set_title(title, fontsize=12, pad=20)
        
        # Add state labels
        for idx, row in continental_us.iterrows():
            centroid = row.geometry.centroid
            ax.annotate(text=row['STUSPS'], 
                       xy=(centroid.x, centroid.y),
                       horizontalalignment='center',
                       fontsize=6)
        
        # Add inset for Alaska and Hawaii if space permits
        if len(columns) <= 2:
            # Alaska inset
            ax_ak = fig.add_axes([ax.get_position().x0, 0.1, 0.1, 0.1])
            alaska = states_data[states_data['NAME'] == 'Alaska']
            if not alaska.empty:
                alaska.plot(column=column, ax=ax_ak, cmap=cmap)
                ax_ak.axis('off')
                ax_ak.set_title('AK', fontsize=8)
            
            # Hawaii inset
            ax_hi = fig.add_axes([ax.get_position().x0, 0.25, 0.1, 0.1])
            hawaii = states_data[states_data['NAME'] == 'Hawaii']
            if not hawaii.empty:
                hawaii.plot(column=column, ax=ax_hi, cmap=cmap)
                ax_hi.axis('off')
                ax_hi.set_title('HI', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 

def create_model_comparison_map(data_df, column, title, save_path, cmap='viridis'):
    """Create side-by-side maps comparing Linear Regression and Random Forest"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
    
    # Load US states shapefile
    states['NAME'] = states['NAME'].apply(lambda x: x.title())
    
    # Merge data with geographic information
    states_data = states.merge(data_df, how='left', left_on='NAME', right_on='State')
    
    # Filter for continental US
    continental_us = states_data[~states_data['NAME'].isin(['Alaska', 'Hawaii'])]
    
    # Linear Regression Map
    continental_us.plot(column=f'Linear_{column}',
                       ax=ax1,
                       legend=True,
                       legend_kwds={'label': column},
                       missing_kwds={'color': 'lightgrey'},
                       cmap=cmap)
    ax1.set_title(f'Linear Regression: {title}', fontsize=16, pad=20)
    ax1.axis('off')
    
    # Random Forest Map
    continental_us.plot(column=f'RF_{column}',
                       ax=ax2,
                       legend=True,
                       legend_kwds={'label': column},
                       missing_kwds={'color': 'lightgrey'},
                       cmap=cmap)
    ax2.set_title(f'Random Forest: {title}', fontsize=16, pad=20)
    ax2.axis('off')
    
    # Add state labels to both maps
    for ax in [ax1, ax2]:
        for idx, row in continental_us.iterrows():
            centroid = row.geometry.centroid
            ax.annotate(text=row['STUSPS'], 
                       xy=(centroid.x, centroid.y),
                       horizontalalignment='center',
                       fontsize=8)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 