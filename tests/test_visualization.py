"""Tests for visualization utilities."""

import pytest
import numpy as np
from pathlib import Path
from src.visualization.plots import plot_predictions, plot_validation_metrics

@pytest.fixture
def sample_predictions():
    """Generate sample predictions for testing."""
    years = np.arange(2020, 2041)
    predictions = {
        'COVID-19': {
            'median': np.linspace(0.4, 0.5, len(years)),
            'lower': np.linspace(0.35, 0.45, len(years)),
            'upper': np.linspace(0.45, 0.55, len(years))
        },
        'Ebola': {
            'median': np.linspace(0.38, 0.48, len(years)),
            'lower': np.linspace(0.33, 0.43, len(years)),
            'upper': np.linspace(0.43, 0.53, len(years))
        },
        'HIV': {
            'median': np.linspace(0.42, 0.52, len(years)),
            'lower': np.linspace(0.37, 0.47, len(years)),
            'upper': np.linspace(0.47, 0.57, len(years))
        }
    }
    return predictions

def test_plot_predictions(sample_predictions, tmp_path):
    """Test prediction plotting."""
    save_path = tmp_path / 'predictions.png'
    plot_predictions(sample_predictions, save_path)
    assert save_path.exists()

def test_plot_validation_metrics(tmp_path):
    """Test validation metrics plotting."""
    metrics = {
        'mae': [0.01, 0.02, 0.015, 0.018],
        'rmse': [0.015, 0.025, 0.02, 0.022],
        'coverage': [0.95, 0.96, 0.94, 0.95]
    }
    save_path = tmp_path / 'metrics.png'
    plot_validation_metrics(metrics, save_path)
    assert save_path.exists()
