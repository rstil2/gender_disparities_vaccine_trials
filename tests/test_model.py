"""Tests for core model functionality."""

import pytest
import numpy as np
import pandas as pd
import pymc as pm
from src.analysis.model import build_model, generate_predictions

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'year': np.random.randint(2010, 2023, n),
        'disease': np.random.choice(['COVID-19', 'Ebola', 'HIV'], n),
        'participants': np.random.poisson(100, n),
    })
    data['females'] = np.random.binomial(data.participants, 0.4)
    return data
    
def test_model_build(sample_data):
    """Test model building functionality."""
    model = build_model(sample_data)
    assert isinstance(model, pm.Model)
    
def test_predictions(sample_data):
    """Test prediction generation."""
    with build_model(sample_data):
        trace = pm.sample(draws=100, tune=100, chains=2)
    
    years = np.arange(2020, 2041)
    preds = generate_predictions(trace, years, ['COVID-19', 'Ebola', 'HIV'])
    
    assert isinstance(preds, dict)
    assert all(d in preds for d in ['COVID-19', 'Ebola', 'HIV'])
    assert all(k in preds['COVID-19'] for k in ['median', 'lower', 'upper'])
