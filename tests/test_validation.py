"""Tests for validation utilities."""

import pytest
import numpy as np
import pandas as pd
from src.validation.cross_val import compute_metrics, temporal_cv

@pytest.fixture
def validation_data():
    """Generate sample data for validation testing."""
    true_values = np.array([0.4, 0.45, 0.5, 0.55, 0.6])
    predictions = np.array([0.42, 0.44, 0.51, 0.53, 0.58])
    ci_lower = predictions - 0.05
    ci_upper = predictions + 0.05
    return true_values, predictions, ci_lower, ci_upper

def test_metric_computation(validation_data):
    """Test metric computation."""
    true_values, predictions, ci_lower, ci_upper = validation_data
    metrics = compute_metrics(true_values, predictions, ci_lower, ci_upper)
    
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'coverage' in metrics
    assert 0 <= metrics['mae'] <= 1
    assert 0 <= metrics['rmse'] <= 1
    assert 0 <= metrics['coverage'] <= 1

def test_temporal_cv(sample_data):
    """Test temporal cross-validation."""
    metrics = temporal_cv(sample_data, n_splits=3)
    
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'coverage' in metrics
    assert len(metrics['mae']) > 0
