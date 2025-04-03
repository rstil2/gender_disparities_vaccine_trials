"""Tests for results handling utilities."""

import pytest
import numpy as np
import arviz as az
from pathlib import Path
from src.analysis.results import Results, summarize_results

@pytest.fixture
def sample_results(tmp_path):
    """Generate sample results for testing."""
    # Create mock trace
    data = np.random.randn(4, 100, 10)
    trace = az.InferenceData(
        posterior=az.convert_to_dataset(data, dims=["chain", "draw", "param"])
    )
    
    predictions = {
        'COVID-19': {
            'median': 0.45,
            'lower': 0.40,
            'upper': 0.50
        },
        'Ebola': {
            'median': 0.43,
            'lower': 0.38,
            'upper': 0.48
        }
    }
    
    metrics = {
        'mae': 0.015,
        'rmse': 0.018,
        'coverage': 0.95
    }
    
    return Results(trace, predictions, metrics)

def test_results_save_load(sample_results, tmp_path):
    """Test saving and loading results."""
    # Save results
    output_dir = tmp_path / 'results'
    sample_results.save_all(output_dir)
    
    # Check files exist
    assert (output_dir / 'trace.nc').exists()
    assert (output_dir / 'results.json').exists()
    assert (output_dir / 'tables' / 'predictions.csv').exists()
    assert (output_dir / 'tables' / 'metrics.csv').exists()
    
    # Load results
    loaded = Results.load(output_dir)
    assert loaded.predictions == sample_results.predictions
    assert loaded.metrics == sample_results.metrics
    
def test_results_summary(sample_results):
    """Test results summarization."""
    summary = summarize_results(sample_results)
    assert isinstance(summary, str)
    assert "Analysis Results Summary" in summary
    assert "Performance Metrics:" in summary
    assert "2040 Predictions:" in summary
