"""Core model implementation for gender disparities analysis."""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, Optional, Tuple

def build_model(
    df: pd.DataFrame,
    coords: Optional[Dict] = None
) -> pm.Model:
    """Build PyMC model for gender participation analysis."""
    if coords is None:
        coords = {"disease": sorted(df.disease.unique())}
        
    with pm.Model(coords=coords) as model:
        # Global parameters
        μ_β0 = pm.Normal("μ_β0", 0, 0.5)
        σ_β0 = pm.HalfNormal("σ_β0", 0.3)
        μ_β1 = pm.Normal("μ_β1", 0, 0.05)
        σ_β1 = pm.HalfNormal("σ_β1", 0.03)
        
        # Disease-specific parameters
        β0_offset = pm.Normal("β0_offset", 0, 1, dims="disease")
        β0 = pm.Deterministic("β0", μ_β0 + β0_offset * σ_β0, dims="disease")
        
        β1_offset = pm.Normal("β1_offset", 0, 1, dims="disease")
        β1 = pm.Deterministic("β1", μ_β1 + β1_offset * σ_β1, dims="disease")
        
        # Linear predictor
        disease_idx = pd.Categorical(df.disease, categories=coords["disease"]).codes
        logit_p = β0[disease_idx] + β1[disease_idx] * (df.year - 2000)
        p = pm.Deterministic("p", pm.math.invlogit(logit_p))
        
        # Likelihood
        y = pm.Binomial("y", n=df.participants, p=p, observed=df.females)
        
    return model

def sample_model(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    return_inferencedata: bool = True,
    random_seed: int = 42
) -> az.InferenceData:
    """Sample from the model using NUTS."""
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=return_inferencedata,
            random_seed=random_seed
        )
    return trace

def generate_predictions(
    trace: az.InferenceData,
    years: np.ndarray,
    diseases: list
) -> Dict[str, np.ndarray]:
    """Generate predictions with uncertainty intervals."""
    predictions = {}
    
    for disease in diseases:
        β0 = trace.posterior.β0.sel(disease=disease)
        β1 = trace.posterior.β1.sel(disease=disease)
        
        logit_p = β0 + β1 * (years - 2000)
        p = 1 / (1 + np.exp(-logit_p))
        
        predictions[disease] = {
            'median': float(p.median()),
            'lower': float(p.quantile(0.025)),
            'upper': float(p.quantile(0.975))
        }
        
    return predictions
