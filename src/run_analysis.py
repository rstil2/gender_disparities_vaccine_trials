"""Main analysis pipeline."""

import argparse
from pathlib import Path
import pandas as pd
from analysis.model import build_model, sample_model, generate_predictions
from analysis.results import Results
from validation.cross_val import temporal_cv
from visualization.plots import plot_predictions, plot_validation_metrics

def run_analysis(data_path: str, output_dir: str):
    """Run complete analysis pipeline."""
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load and process data
    df = pd.read_csv(data_path)
    
    # Validation
    cv_metrics = temporal_cv(df)
    plot_validation_metrics(
        cv_metrics,
        output_dir / 'figures' / 'validation_metrics.png'
    )
    
    # Full model
    model = build_model(df)
    trace = sample_model(model)
    
    # Predictions
    predictions = generate_predictions(
        trace,
        years=range(2020, 2041),
        diseases=df.disease.unique()
    )
    
    # Create and save results
    results = Results(
        trace=trace,
        predictions=predictions,
        metrics={
            'mae': np.mean(cv_metrics['mae']),
            'rmse': np.mean(cv_metrics['rmse']),
            'coverage': np.mean(cv_metrics['coverage'])
        }
    )
    
    results.save_all(output_dir)
    
    # Generate visualizations
    plot_predictions(
        predictions,
        output_dir / 'figures' / 'predictions.png'
    )
    
    # Print summary
    print("\nAnalysis complete. Results saved to:", output_dir)
    print("\n", summarize_results(results))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to input data")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    args = parser.parse_args()
    
    run_analysis(args.data_path, args.output_dir)

def get_config():
    """Get analysis configuration."""
    return {
        "model": {
            "draws": 2000,
            "chains": 4,
            "target_accept": 0.95
        },
        "data": {
            "start_year": 1990,
            "end_year": 2023
        },
        "analysis": {
            "cv_folds": 5,
            "prediction_horizon": 2040
        }
    }

if __name__ == "__main__":
    parser.add_argument(
        "--generate-docs",
        action="store_true",
        help="Generate reproducible research documentation"
    )
    args = parser.parse_args()
    
    # Run analysis
    results = run_analysis(args.data_path, args.output_dir)
    
    # Generate documentation if requested
    if args.generate_docs:
        from analysis.reproducible import ResearchCompanion
        companion = ResearchCompanion(
            results=results,
            config=get_config(),
            output_dir=Path(args.output_dir) / "documentation"
        )
        companion.generate_all()
