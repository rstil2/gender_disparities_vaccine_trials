"""Generate figure comparing trial participation rates with US Census data."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, List
import random

from src.visualization.plots import plot_census_comparison

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def bootstrap_confidence_interval(data: List[float], n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval using bootstrapping.
    
    Args:
        data: List of values to bootstrap
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower, upper) bounds of the confidence interval
    """
    if len(data) < 2:
        # Can't bootstrap with just one sample, return the value with a small range
        if len(data) == 1:
            return (data[0] - 0.02, data[0] + 0.02)
        return (0, 0)  # No data
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = random.choices(data, k=len(data))
        bootstrap_means.append(np.mean(sample))
    
    # Calculate confidence interval
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return (lower_bound, upper_bound)

def load_trial_data(data_path: Path) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, List[float]], Dict[str, Tuple[float, float]]]:
    """Load and process trial participation data.
    
    Args:
        data_path: Path to trial participation CSV
        
    Returns:
        Tuple containing:
        - participation_rates: Dict mapping disease to average female participation rate
        - trial_counts: Dict mapping disease to total participant count
        - individual_trials: Dict mapping disease to list of individual trial percentages
        - confidence_intervals: Dict mapping disease to (lower, upper) confidence interval
    """
    df = pd.read_csv(data_path)
    
    # Normalize percentages (convert to proportion if given as percentage)
    df['female_percentage_normalized'] = df['female_percentage'].apply(
        lambda x: x / 100 if x > 1 else x
    )
    
    # Validate percentages are in the expected range (0-1)
    invalid_rows = df[(df['female_percentage_normalized'] < 0) | (df['female_percentage_normalized'] > 1)]
    if not invalid_rows.empty:
        logging.warning(f"Found {len(invalid_rows)} rows with invalid percentages outside range [0,1]. Clipping values.")
        df['female_percentage_normalized'] = df['female_percentage_normalized'].clip(0, 1)
    
    # Aggregate participation rates by disease
    participation_rates = df.groupby('disease')['female_percentage_normalized'].mean().to_dict()
    
    # Get total participant counts by disease
    trial_counts = df.groupby('disease')['total_participants'].sum().to_dict()
    
    # Get individual trial percentages for each disease
    individual_trials = {}
    for disease in df['disease'].unique():
        individual_trials[disease] = df[df['disease'] == disease]['female_percentage_normalized'].tolist()
    
    # Calculate confidence intervals using bootstrapping
    confidence_intervals = {}
    for disease, trials in individual_trials.items():
        confidence_intervals[disease] = bootstrap_confidence_interval(trials)
    
    logging.info(f"Processed {len(df)} trial data rows across {len(df['disease'].unique())} diseases")
    return participation_rates, trial_counts, individual_trials, confidence_intervals

def load_census_data(data_path: Path) -> Dict[str, float]:
    """Load processed census data.
    
    Args:
        data_path: Path to census data CSV
        
    Returns:
        Dictionary mapping year (as string) to female percentage as proportion (0-1)
    """
    df = pd.read_csv(data_path)
    
    # Normalize percentages (convert to proportion if given as percentage)
    df['female_percentage_normalized'] = df['female_percentage'].apply(
        lambda x: x / 100 if x > 1 else x
    )
    
    # Validate percentages are in the expected range (0-1)
    invalid_rows = df[(df['female_percentage_normalized'] < 0) | (df['female_percentage_normalized'] > 1)]
    if not invalid_rows.empty:
        logging.warning(f"Found {len(invalid_rows)} census rows with invalid percentages outside range [0,1]. Clipping values.")
        df['female_percentage_normalized'] = df['female_percentage_normalized'].clip(0, 1)
    
    # Convert years to strings to ensure compatibility with the plot function
    return dict(zip(df['year'].astype(str), df['female_percentage_normalized']))

def main():
    """Main function to generate comparison figure."""
    setup_logging()
    
    # Set up paths
    data_dir = Path('data')
    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)
    
    try:
        # Load trial data with extended information
        trial_data, trial_counts, individual_trials, confidence_intervals = load_trial_data(
            data_dir / 'processed' / 'trial_participation.csv'
        )
        
        # Log information about the data
        # Log information about the data
        logging.info(f"Loaded data for {len(trial_data)} diseases")
        for disease, rate in trial_data.items():
            ci_lower, ci_upper = confidence_intervals[disease]
            logging.info(f"{disease}: {rate*100:.1f}% female participation [{ci_lower*100:.1f}%-{ci_upper*100:.1f}%], n={trial_counts[disease]:,}")
        # Load census data
        census_data = load_census_data(data_dir / 'census' / 'processed_census_data.csv')
        
        # Generate comparison figure
        output_path = figures_dir / 'Figure2_census_comparison.png'
        plot_census_comparison(
            trial_data=trial_data,
            census_data=census_data,
            trial_counts=trial_counts,
            individual_trials=individual_trials,
            confidence_intervals=confidence_intervals,
            save_path=output_path
        )
        
        logging.info(f"Successfully generated census comparison figure: {output_path}")
        
    except Exception as e:
        logging.error(f"Error generating census comparison figure: {str(e)}")
        raise

if __name__ == '__main__':
    main()

