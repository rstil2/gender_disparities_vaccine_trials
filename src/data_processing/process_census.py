"""Process US Census data for gender comparison analysis."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_census_data(years: list) -> pd.DataFrame:
    """Fetch or load US Census data for specified years.
    
    Args:
        years: List of years to fetch data for
        
    Returns:
        DataFrame containing census data with columns 'year' and 'female_percentage'
    """
    # For now, we'll use approximate values based on recent census data
    # In practice, you would fetch this from the Census API
    data = {
        'year': years,
        'female_percentage': [50.8, 50.8, 50.7, 50.7, 50.6]  # Example values
    }
    return pd.DataFrame(data)

def process_census_data(df: pd.DataFrame) -> Dict[int, float]:
    """Process census data into required format.
    
    Args:
        df: DataFrame containing census data
        
    Returns:
        Dictionary mapping years to female percentages
    """
    return dict(zip(df['year'], df['female_percentage']))

def save_census_data(data: Dict[int, float], output_path: Path):
    """Save processed census data.
    
    Args:
        data: Dictionary mapping years to female percentages
        output_path: Path where to save the CSV file
    """
    df = pd.DataFrame(list(data.items()), columns=['year', 'female_percentage'])
    df.to_csv(output_path, index=False)
    logging.info(f"Saved census data to {output_path}")

def main():
    """Main function to process census data."""
    setup_logging()
    
    # Create output directory if it doesn't exist
    output_dir = Path('data/census')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process data for years 2020-2024
    years = list(range(2020, 2025))
    
    try:
        # Get and process census data
        df = get_census_data(years)
        processed_data = process_census_data(df)
        
        # Save processed data
        output_path = output_dir / 'processed_census_data.csv'
        save_census_data(processed_data, output_path)
        
        logging.info("Census data processing completed successfully")
        
    except Exception as e:
        logging.error(f"Error processing census data: {str(e)}")
        raise

if __name__ == '__main__':
    main()

