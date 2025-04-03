"""Visualization utilities for analysis results."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Tuple

def set_style():
    """Set consistent plotting style."""
    plt.style.use('default')  # Use default style instead of seaborn
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    
def plot_predictions(
    predictions: Dict[str, Dict[str, float]],
    save_path: Path,
    years: Optional[np.ndarray] = None,
    figsize: tuple = (12, 8)
):
    """Plot predictions with uncertainty intervals."""
    if years is None:
        years = np.arange(2020, 2041)
        
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {
        'COVID-19': '#1f77b4',
        'Ebola': '#ff7f0e',
        'HIV': '#2ca02c'
    }
    
    for disease, color in colors.items():
        pred = predictions[disease]
        ax.plot(years, pred['median'], color=color, label=disease, lw=2)
        ax.fill_between(
            years,
            pred['lower'],
            pred['upper'],
            color=color,
            alpha=0.2
        )
        
    ax.axhline(0.5, color='r', linestyle='--', label='Gender Parity')
    ax.set_xlabel('Year')
    ax.set_ylabel('Female Participation Rate')
    ax.set_title('Projected Female Participation Rates with 95% CI')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_validation_metrics(
    metrics: Dict[str, List[float]],
    save_path: Path,
    figsize: tuple = (15, 5)
):
    """Plot validation metrics across folds."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for ax, (metric, values) in zip(axes, metrics.items()):
        sns.boxplot(y=values, ax=ax)
        ax.set_title(f'{metric.upper()}')
        if metric == 'coverage':
            ax.axhline(0.95, color='r', linestyle='--', alpha=0.5)
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_sensitivity_analysis(
    results: Dict[str, Dict[str, float]],
    save_path: Path,
    figsize: tuple = (10, 6)
):
    """Plot sensitivity analysis results."""
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    scales = [float(k.split('_')[1]) for k in results.keys()]
    means = [v['mean'] for v in results.values()]
    ci_widths = [v['ci_width'] for v in results.values()]
    
    ax.errorbar(
        scales,
        means,
        yerr=np.array(ci_widths)/2,
        fmt='o-',
        capsize=5
    )
    
    ax.set_xlabel('Prior Scale')
    ax.set_ylabel('2040 Prediction')
    ax.set_title('Sensitivity to Prior Specification')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_census_comparison(
    trial_data: Dict[str, float],
    census_data: Dict[str, float],
    save_path: Path,
    trial_counts: Dict[str, int],
    individual_trials: Dict[str, List[float]],
    confidence_intervals: Dict[str, Tuple[float, float]],
    figsize: tuple = (12, 7)
):
    """Create bar plot comparing trial participation rates with US Census data.
    
    Args:
        trial_data: Dictionary with disease names as keys and female participation rates as values
        census_data: Dictionary with years as keys and female population percentages as values
        save_path: Path where to save the figure
        trial_counts: Dictionary with disease names as keys and total participant counts as values
        individual_trials: Dictionary with disease names as keys and lists of individual trial percentages as values
        confidence_intervals: Dictionary with disease names as keys and tuples of (lower, upper) 95% CI as values
        figsize: Tuple specifying figure dimensions
    """
    # Dictionary mapping each disease to its respective census year
    disease_to_year = {
        'COVID-19': '2020',  # COVID-19 pandemic peak was in 2020
        'Ebola': '2014',     # Ebola outbreak was primarily in 2014
        'HIV': '2010'        # Using 2010 for HIV for historical comparison
    }
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # We no longer use an average census percentage, but specific values per disease
    
    # Sort diseases by participation rate for better visualization
    sorted_diseases = sorted(trial_data.items(), key=lambda x: x[1])
    diseases = [d[0] for d in sorted_diseases]
    participation_rates = [d[1] for d in sorted_diseases]
    
    # Plot the bars
    bar_width = 0.4
    x = np.arange(len(diseases))
    
    # Create bars for trial data
    # Calculate error bar heights correctly for both upper and lower bounds
    yerr = np.array([
        [participation_rates[i] - confidence_intervals[diseases[i]][0] for i in range(len(diseases))],  # lower errors
        [confidence_intervals[diseases[i]][1] - participation_rates[i] for i in range(len(diseases))]   # upper errors
    ])
    
    # Increase bar width for better visibility
    bar_width = 0.6
    
    trial_bars = ax.bar(
        x,
        participation_rates,
        width=bar_width,
        color='#1f77b4',
        label='Clinical Trials',
        yerr=yerr,
        capsize=8,
        error_kw={'ecolor': 'black', 'linewidth': 1.5, 'capthick': 1.5, 'capsize': 5}
    )
    
    # Draw horizontal lines for census data for each year instead of a single average
    census_lines = []
    
    # Add a legend entry for the census data (will be replaced with actual lines below)
    census_legend_line = ax.plot([], [], color='#ff7f0e', linestyle='-', linewidth=2.5, label='US Census Data')[0]
    
    # Add text annotations for census years
    legend_entries = []
    for year, value in census_data.items():
        legend_text = f"{year} Census: {value:.1%}"
        legend_entries.append(legend_text)
    
    # Add statistical significance markers
    for i, (disease, rate) in enumerate(zip(diseases, participation_rates)):
        # Perform chi-square test for proportions instead of t-test
        # This is more appropriate for comparing proportions/percentages
        
        # Get number of participants for this disease
        n_participants = trial_counts[disease]
        
        # Calculate observed counts
        n_female_observed = int(rate * n_participants)
        n_male_observed = n_participants - n_female_observed
        
        # Get the relevant census year for this disease
        relevant_year = disease_to_year[disease]
        relevant_census_pct = census_data[relevant_year]
        
        # Calculate expected counts based on disease-specific census data
        n_female_expected = int(relevant_census_pct * n_participants)
        n_male_expected = n_participants - n_female_expected
        
        # Draw the horizontal line for this disease's census data
        census_x_start = i - bar_width/2
        census_x_end = i + bar_width/2
        census_line = ax.plot(
            [census_x_start, census_x_end],
            [relevant_census_pct, relevant_census_pct],
            color='#ff7f0e',
            linestyle='-',
            linewidth=2.5
        )
        
        # Create the observed and expected frequency tables
        observed = np.array([n_female_observed, n_male_observed])
        expected = np.array([n_female_expected, n_male_expected])
        
        # Print the contingency table for debugging
        print(f"\nContingency table for {disease} (compared to {relevant_year} census):")
        contingency_df = pd.DataFrame({
            'Observed': [n_female_observed, n_male_observed],
            'Expected': [n_female_expected, n_male_expected],
            'Difference': [n_female_observed - n_female_expected, n_male_observed - n_male_expected]
        }, index=['Female', 'Male'])
        print(contingency_df)
        
        # Perform chi-square test
        chi2, p_value, _, _ = stats.chi2_contingency(
            np.array([[n_female_observed, n_female_expected], 
                     [n_male_observed, n_male_expected]])
        )
        
        # Print actual p-values for each disease
        print(f"Statistical significance for {disease}:")
        print(f"  - Female participation rate: {rate:.1%}")
        print(f"  - {relevant_year} Census: {relevant_census_pct:.1%}")
        print(f"  - Difference: {(rate - relevant_census_pct):.1%}")
        print(f"  - Female counts: {n_female_observed} observed vs {n_female_expected} expected")
        print(f"  - Chi-square statistic: {chi2:.4f}")
        print(f"  - p-value: {p_value:.6f}")
        
        # Add significance markers
        significance = ""
        if p_value < 0.001:
            significance = "***"
            sig_level = "p<0.001"
        elif p_value < 0.01:
            significance = "**"
            sig_level = "p<0.01"
        elif p_value < 0.05:
            significance = "*"
            sig_level = "p<0.05"
        else:
            sig_level = "not significant"
            
        # Log significance level for each disease
        print(f"  - Significance: {sig_level} {significance}")
            
        if significance:
            # Position stars well above the error bars with more space
            # Calculate maximum height needed for the stars
            y_pos = rate + yerr[1][i] + 0.07  # Increased offset for better visibility
            ax.text(i, y_pos, significance, ha='center', va='bottom', fontsize=20, 
                   fontweight='bold', color='#d62728')  # Make stars even larger
            
    # Add participant counts above each bar
    for i, (disease, rate) in enumerate(zip(diseases, participation_rates)):
        count = trial_counts[disease]
        # Position the count text between the top of the bar and the significance stars
        # to avoid overlap with either
        offset = yerr[1][i] + 0.025  # Adjusted to make room for stars above
        ax.text(i, rate + offset, f"n={count:,}", 
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='#444444')
                
        # Also add the actual percentage value on the bar for clarity
        ax.text(i, rate / 2, f"{rate:.1%}", 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        # Add census year-specific annotation near the census line
        relevant_year = disease_to_year[disease]
        relevant_census_pct = census_data[relevant_year]
        ax.text(i, relevant_census_pct + 0.02, f"{relevant_year}: {relevant_census_pct:.1%}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#ff7f0e')
    
    
    
    # Customize the plot
    ax.set_xlabel('Disease', fontsize=14, fontweight='bold')
    ax.set_ylabel('Female Participation Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Clinical Trial Female Participation vs. US Census by Outbreak Year', fontsize=16, fontweight='bold')
    ax.set_xticklabels(diseases, rotation=0, ha='center', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add a line for gender parity (50%)
    ax.axhline(0.5, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Gender Parity')
    
    # Set y-axis limits with some padding
    # Set y-axis limits with more padding for the annotations and significance stars
    max_value = max(max(participation_rates) + max(yerr[1]), avg_census_pct)
    ax.set_ylim(0, max_value * 1.5)  # Increased from 1.4 to 1.5 to ensure stars are more visible
    
    # Add connecting lines from bars to significance stars for clarity
    for i, (disease, rate) in enumerate(zip(diseases, participation_rates)):
        # Re-calculate p-value for line drawing using same chi-square method
        n_participants = trial_counts[disease]
        n_female_observed = int(rate * n_participants)
        n_male_observed = n_participants - n_female_observed
        
        # Get the relevant census year for this disease
        relevant_year = disease_to_year[disease]
        relevant_census_pct = census_data[relevant_year]
        
        # Calculate expected counts based on disease-specific census data
        n_female_expected = int(relevant_census_pct * n_participants)
        n_male_expected = n_participants - n_female_expected
        
        chi2, p_value, _, _ = stats.chi2_contingency(
            np.array([[n_female_observed, n_female_expected], 
                     [n_male_observed, n_male_expected]])
        )
            
        if p_value < 0.05:  # Only draw for significant results
            y_error_top = rate + yerr[1][i]
            y_star_pos = y_error_top + 0.07
            # Add a subtle connecting line
            ax.plot([i, i], [y_error_top + 0.01, y_star_pos - 0.01], 
                   color='#d62728', alpha=0.5, linestyle=':')
            # Add a small marker at the bottom of the star
            ax.scatter(i, y_star_pos - 0.01, color='#d62728', s=20, alpha=0.7)
    
    # Format y-axis ticks as percentages
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    
    # Add text explaining significance - moved to a more prominent position
    # Add text explaining significance - moved to a more prominent position
    # and made more visible with a background
    text_box = ax.text(0.02, 0.97, 
             "* p<0.05, ** p<0.01, *** p<0.001", 
             ha='left', va='top', fontsize=14, fontweight='bold',
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#d62728', boxstyle='round,pad=0.5'))
             
    # Print summary of significance testing
    print("\nSummary of significance testing:")
    for disease, rate in zip(diseases, participation_rates):
        n_participants = trial_counts[disease]
        n_female_observed = int(rate * n_participants)
        n_male_observed = n_participants - n_female_observed
        
        # Get the relevant census year for this disease
        relevant_year = disease_to_year[disease]
        relevant_census_pct = census_data[relevant_year]
        
        # Calculate expected counts based on disease-specific census data
        n_female_expected = int(relevant_census_pct * n_participants)
        n_male_expected = n_participants - n_female_expected
        
        chi2, p_value, _, _ = stats.chi2_contingency(
            np.array([[n_female_observed, n_female_expected], 
                     [n_male_observed, n_male_expected]])
        )
    # Ensure enough space for all elements
    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
