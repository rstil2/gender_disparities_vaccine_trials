"""Results processing and export utilities."""

import json
import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class Results:
    """Container for analysis results with export capabilities."""
    
    def __init__(
        self,
        trace: az.InferenceData,
        predictions: Dict[str, Dict[str, float]],
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.trace = trace
        self.predictions = predictions
        self.metrics = metrics
        self.metadata = metadata or {
            'timestamp': datetime.now().isoformat(),
            'arviz_version': az.__version__
        }
        
    def to_json(self, path: Path) -> None:
        """Export results to JSON."""
        output = {
            'metadata': self.metadata,
            'predictions': self.predictions,
            'metrics': self.metrics
        }
        
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
            
    def to_netcdf(self, path: Path) -> None:
        """Export trace to netCDF format."""
        self.trace.to_netcdf(path)
        
    def to_tables(self, directory: Path) -> None:
        """Export results as CSV tables."""
        directory.mkdir(exist_ok=True)
        
        # Export predictions
        pred_df = pd.DataFrame(self.predictions).round(3)
        pred_df.to_csv(directory / 'predictions.csv')
        
        # Export metrics
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv(directory / 'metrics.csv', index=False)
        
    def save_all(self, output_dir: Path) -> None:
        """Save all results in standard format."""
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save trace
        self.to_netcdf(output_dir / 'trace.nc')
        
        # Save other results
        self.to_json(output_dir / 'results.json')
        self.to_tables(output_dir / 'tables')
        
    @classmethod
    def load(cls, directory: Path) -> 'Results':
        """Load results from directory."""
        # Load trace
        trace = az.from_netcdf(directory / 'trace.nc')
        
        # Load other results
        with open(directory / 'results.json', 'r') as f:
            data = json.load(f)
            
        return cls(
            trace=trace,
            predictions=data['predictions'],
            metrics=data['metrics'],
            metadata=data['metadata']
        )

def summarize_results(results: Results) -> str:
    """Generate a human-readable summary of results."""
    summary = ["Analysis Results Summary", "=" * 20, ""]
    
    # Metrics summary
    summary.append("Performance Metrics:")
    summary.append("-" * 18)
    for metric, value in results.metrics.items():
        summary.append(f"{metric}: {value:.3f}")
    summary.append("")
    
    # Predictions summary
    summary.append("2040 Predictions:")
    summary.append("-" * 15)
    for disease, pred in results.predictions.items():
        summary.append(
            f"{disease}: {pred['median']:.1%} ({pred['lower']:.1%} - {pred['upper']:.1%})"
        )
    summary.append("")
    
    # Metadata
    summary.append("Analysis Metadata:")
    summary.append("-" * 16)
    for key, value in results.metadata.items():
        summary.append(f"{key}: {value}")
        
    return "\n".join(summary)
