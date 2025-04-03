"""Cross-validation and model validation utilities."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
from ..analysis.model import build_model, sample_model

def compute_metrics(
    true_values: np.ndarray,
    predictions: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray
) -> Dict[str, float]:
    """Compute validation metrics."""
    mae = np.mean(np.abs(predictions - true_values))
    rmse = np.sqrt(np.mean((predictions - true_values) ** 2))
    coverage = np.mean((true_values >= ci_lower) & (true_values <= ci_upper))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'coverage': coverage
    }

def temporal_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: int = 3
) -> Dict[str, List[float]]:
    """Perform temporal cross-validation."""
    cv = TimeSeriesSplit(n_splits=n_splits)
    metrics = {
        'mae': [],
        'rmse': [],
        'coverage': []
    }
    
    for train_idx, test_idx in cv.split(df):
        # Ensure minimum training size
        if len(train_idx) < min_train_size:
            continue
            
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        
        # Fit model
        model = build_model(train_data)
        trace = sample_model(model, draws=1000, tune=1000)
        
        # Generate predictions
        preds = trace.posterior.p.mean(['chain', 'draw'])
        ci_lower = trace.posterior.p.quantile(0.025, ['chain', 'draw'])
        ci_upper = trace.posterior.p.quantile(0.975, ['chain', 'draw'])
        
        # Compute metrics
        fold_metrics = compute_metrics(
            test_data.females / test_data.participants,
            preds,
            ci_lower,
            ci_upper
        )
        
        for metric, value in fold_metrics.items():
            metrics[metric].append(value)
            
    return metrics

def sensitivity_analysis(
    df: pd.DataFrame,
    prior_scales: List[float] = [0.5, 1.0, 2.0]
) -> Dict[str, Dict[str, float]]:
    """Perform prior sensitivity analysis."""
    results = {}
    
    for scale in prior_scales:
        model = build_model(
            df,
            prior_scale=scale  # Need to modify build_model to accept this
        )
        trace = sample_model(model)
        
        # Extract 2040 predictions
        pred_2040 = trace.posterior.p.sel(year=2040).mean(['chain', 'draw'])
        ci_lower = trace.posterior.p.sel(year=2040).quantile(0.025, ['chain', 'draw'])
        ci_upper = trace.posterior.p.sel(year=2040).quantile(0.975, ['chain', 'draw'])
        
        results[f'scale_{scale}'] = {
            'mean': float(pred_2040),
            'ci_width': float(ci_upper - ci_lower)
        }
        
    return results
