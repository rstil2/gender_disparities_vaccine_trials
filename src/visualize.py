import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from datetime import datetime

def create_visualization(model_df, predictions, diseases):
    """Create visualizations for disease prevalence trends."""
    print("\nCreating visualization...")
    
    # Create figure and axes
    plt.clf()  # Clear any existing plots
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define color-blind friendly colors
    colors = {
        'COVID-19': '#0077BB',    # Blue
        'Ebola': '#FFDD00',       # Yellow
        'HIV': '#CC0000'          # Red
    }
    
    # Current year for prediction split
    current_year = datetime.now().year
    max_year = current_year  # Initialize with current year
    min_year = model_df['year'].min()
    
    # Plot data for each disease
    for disease in diseases:
        print(f"\nProcessing {disease}:")
        
        # Plot historical data
        disease_data = model_df[model_df['disease'] == disease]
        real_data = disease_data[~disease_data['is_synthetic']]
        
        if len(real_data) > 0:
            years = real_data['year'].values
            rates = real_data['prevalence'].values  # Already in percentages
            print(f"Historical data points: {list(zip(years, rates))}")
            
            ax.scatter(years, rates,
                    color=colors[disease],
                    marker='o', s=100, zorder=4,
                    label=f"{disease} (Historical)")
            
            if len(real_data) > 1:
                ax.plot(years, rates,
                      color=colors[disease], linestyle='-', linewidth=2,
                      zorder=3)
        
        # Process predictions
        pred_years = np.array(predictions[disease]['years'])
        pred_mean = np.array(predictions[disease]['mean']) * 100  # Convert to percentage
        pred_lower = np.array(predictions[disease]['lower']) * 100
        pred_upper = np.array(predictions[disease]['upper']) * 100
        
        print(f"Prediction range: {pred_mean.min():.1f}% - {pred_mean.max():.1f}%")
        
        max_year = max(max_year, pred_years.max())
        min_year = min(min_year, pred_years.min())
        
        # Plot predictions
        future_mask = pred_years > current_year
        if np.any(future_mask):
            future_years = pred_years[future_mask]
            future_mean = pred_mean[future_mask]
            future_lower = pred_lower[future_mask]
            future_upper = pred_upper[future_mask]
            
            ax.plot(future_years, future_mean,
                   color=colors[disease], linestyle=':', linewidth=2,
                   alpha=0.8, zorder=2, label=f"{disease} (Projected)")
            
            ax.fill_between(future_years, future_lower, future_upper,
                           color=colors[disease], alpha=0.1, zorder=1)
    
    # Add vertical line at current year
    ax.axvline(x=current_year, color='gray', linestyle='--', alpha=0.7)
    ax.text(current_year + 0.5, 95, 'Predictions →', 
            fontsize=10, alpha=0.7)
    
    # Add parity line
    ax.axhline(y=50, color='black', linestyle='-.', 
              linewidth=1.5, label='Gender Parity', zorder=1)
    
    # Set axis limits with padding
    x_padding = 1
    ax.set_xlim(min_year - x_padding, max_year + x_padding)
    ax.set_ylim(0, 100)
    
    # Customize the plot
    ax.set_title('Female Participation in Clinical Trials\nHistorical Data and Projections', 
            fontsize=16, pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Female Participation (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('participation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to 'participation_analysis.png'")

if __name__ == "__main__":
    print("This module provides visualization functions for disease prediction models.")
    print("Import and use the create_visualization function with your model data.")
