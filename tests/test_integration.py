"""Integration tests for complete analysis pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.analysis.model import build_model, sample_model, generate_predictions
from src.validation.cross_val import temporal_cv
from src.analysis.results import Results
from src.analysis.reproducible import ResearchCompanion

@pytest.fixture
def sample_trial_data(tmp_path):
    """Generate sample trial data for integration testing."""
    np.random.seed(42)
    n = 100
    
    # Generate realistic trial data
    data = pd.DataFrame({
        'year': np.random.randint(1990, 2024, n),
        'disease': np.random.choice(
            ['COVID-19', 'Ebola', 'HIV'],
            n,
            p=[0.4, 0.3, 0.3]
        ),
        'participants': np.random.poisson(100, n),
        'phase': np.random.choice([1, 2, 3], n),
        'funding': np.random.choice(
            ['Industry', 'Public', 'Non-profit'],
            n
        ),
        'region': np.random.choice(['LMIC', 'High-Income'], n)
    })
    
    # Generate female participants with realistic patterns
    base_rates = {
        'COVID-19': 0.4,
        'Ebola': 0.38,
        'HIV': 0.42
    }
    
    data['females'] = [
        np.random.binomial(
            row.participants,
            base_rates[row.disease] + 0.001 * (row.year - 2000)
        )
        for row in data.itertuples()
    ]
    
    # Save to CSV
    data_path = tmp_path / 'trial_data.csv'
    data.to_csv(data_path, index=False)
    
    return data_path

@pytest.fixture
def analysis_config():
    """Sample analysis configuration."""
    return {
        'model': {
            'draws': 100,  # Reduced for testing
            'chains': 2,
            'target_accept': 0.95
        },
        'data': {
            'start_year': 1990,
            'end_year': 2023
        },
        'analysis': {
            'cv_folds': 3,
            'prediction_horizon': 2040
        }
    }

def test_full_pipeline(sample_trial_data, analysis_config, tmp_path):
    """Test complete analysis pipeline."""
    # Load data
    df = pd.read_csv(sample_trial_data)
    assert len(df) > 0
    
    # Cross-validation
    cv_metrics = temporal_cv(df, n_splits=analysis_config['analysis']['cv_folds'])
    assert all(k in cv_metrics for k in ['mae', 'rmse', 'coverage'])
    assert all(len(v) > 0 for v in cv_metrics.values())
    
    # Model fitting
    model = build_model(df)
    trace = sample_model(
        model,
        draws=analysis_config['model']['draws'],
        chains=analysis_config['model']['chains']
    )
    assert trace is not None
    
    # Predictions
    predictions = generate_predictions(
        trace,
        years=range(2020, 2041),
        diseases=df.disease.unique()
    )
    assert all(d in predictions for d in ['COVID-19', 'Ebola', 'HIV'])
    
    # Results handling
    results = Results(
        trace=trace,
        predictions=predictions,
        metrics={
            'mae': np.mean(cv_metrics['mae']),
            'rmse': np.mean(cv_metrics['rmse']),
            'coverage': np.mean(cv_metrics['coverage'])
        }
    )
    
    # Save results
    output_dir = tmp_path / 'results'
    results.save_all(output_dir)
    assert (output_dir / 'trace.nc').exists()
    assert (output_dir / 'results.json').exists()
    
    # Generate documentation
    companion = ResearchCompanion(
        results=results,
        config=analysis_config,
        output_dir=output_dir / 'documentation'
    )
    companion.generate_all()
    
    # Verify documentation
    doc_dir = output_dir / 'documentation'
    assert (doc_dir / 'analysis_report.md').exists()
    assert (doc_dir / 'explore_results.ipynb').exists()
    assert (doc_dir / 'diagnostics').exists()
    assert (doc_dir / 'analysis_config.yml').exists()

def test_reproducibility(sample_trial_data, analysis_config, tmp_path):
    """Test analysis reproducibility with fixed seed."""
    np.random.seed(42)
    
    # Run analysis twice
    def run_once():
        df = pd.read_csv(sample_trial_data)
        model = build_model(df)
        trace = sample_model(
            model,
            draws=50,  # Small number for testing
            chains=2,
            random_seed=42
        )
        return trace
        
    trace1 = run_once()
    trace2 = run_once()
    
    # Compare results
    np.testing.assert_allclose(
        trace1.posterior.mean(),
        trace2.posterior.mean(),
        rtol=1e-5
    )
