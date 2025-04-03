import pandas as pd
import numpy as np
import pymc as pm
from sklearn.model_selection import KFold
from datetime import datetime

def fit_model(train_data, diseases):
    """Fit the hierarchical model on training data"""
    with pm.Model() as model:
        # Global parameters
        mu_b0 = pm.Normal("mu_b0", mu=0, sigma=1)
        sigma_b0 = pm.HalfNormal("sigma_b0", sigma=1)
        mu_b1 = pm.Normal("mu_b1", mu=0, sigma=0.1)
        sigma_b1 = pm.HalfNormal("sigma_b1", sigma=0.1)
        
        # Disease-specific parameters
        b0_offset = pm.Normal("b0_offset", mu=0, sigma=1, shape=len(diseases))
        b1_offset = pm.Normal("b1_offset", mu=0, sigma=0.1, shape=len(diseases))
        
        # Calculate rates
        for i, disease in enumerate(diseases):
            disease_data = train_data[train_data['disease'] == disease]
            years = disease_data['year'].values - 2000  # Center years
            
            b0 = mu_b0 + b0_offset[i] * sigma_b0
            b1 = mu_b1 + b1_offset[i] * sigma_b1
            
            logit_p = b0 + b1 * years
            p = pm.math.invlogit(logit_p)
            
            # Likelihood
            pm.Normal(f"obs_{disease}",
                    mu=p,
                    sigma=0.1,
                    observed=disease_data['prevalence'].values / 100)
        
        # Sampling
        trace = pm.sample(1000, tune=500, target_accept=0.9)
    
    return trace

def predict(trace, years, disease_idx, n_samples=1000):
    """Generate predictions from the model"""
    # Extract posterior samples
    b0_samples = trace.posterior['mu_b0'].values.flatten() + \
            trace.posterior['b0_offset'].values[:,:,disease_idx].flatten() * \
            trace.posterior['sigma_b0'].values.flatten()
    
    b1_samples = trace.posterior['mu_b1'].values.flatten() + \
            trace.posterior['b1_offset'].values[:,:,disease_idx].flatten() * \
            trace.posterior['sigma_b1'].values.flatten()
    
    # Generate predictions
    centered_years = years - 2000
    predictions = []
    
    for b0, b1 in zip(b0_samples[:n_samples], b1_samples[:n_samples]):
        logit_p = b0 + b1 * centered_years
        p = 1 / (1 + np.exp(-logit_p))
        predictions.append(p)
    
    predictions = np.array(predictions)
    
    return {
        'mean': np.mean(predictions, axis=0) * 100,
        'lower': np.percentile(predictions, 2.5, axis=0) * 100,
        'upper': np.percentile(predictions, 97.5, axis=0) * 100
    }

def cross_validate(data_df, n_splits=5):
    """Perform k-fold cross-validation"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    diseases = data_df['disease'].unique()
    
    metrics = {
        'mae': [],
        'rmse': [],
        'coverage': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_df)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        train_data = data_df.iloc[train_idx]
        test_data = data_df.iloc[test_idx]
        
        # Fit model on training data
        trace = fit_model(train_data, diseases)
        
        # Evaluate on test data
        fold_mae = []
        fold_rmse = []
        fold_coverage = []
        
        for disease_idx, disease in enumerate(diseases):
            test_disease = test_data[test_data['disease'] == disease]
            if len(test_disease) == 0:
                continue
                
            years = test_disease['year'].values
            true_values = test_disease['prevalence'].values
            
            # Generate predictions
            preds = predict(trace, years, disease_idx)
            
            # Calculate metrics
            mae = np.mean(np.abs(preds['mean'] - true_values))
            rmse = np.sqrt(np.mean((preds['mean'] - true_values)**2))
            coverage = np.mean((true_values >= preds['lower']) & 
                             (true_values <= preds['upper']))
            
            fold_mae.append(mae)
            fold_rmse.append(rmse)
            fold_coverage.append(coverage)
        
        # Store fold metrics
        metrics['mae'].append(np.mean(fold_mae))
        metrics['rmse'].append(np.mean(fold_rmse))
        metrics['coverage'].append(np.mean(fold_coverage))
        
        print(f"Fold metrics:")
        print(f"  MAE: {metrics['mae'][-1]:.2f}%")
        print(f"  RMSE: {metrics['rmse'][-1]:.2f}%")
        print(f"  95% CI Coverage: {metrics['coverage'][-1]*100:.1f}%")
    
    # Calculate overall metrics
    print("\nOverall cross-validation metrics:")
    print(f"MAE: {np.mean(metrics['mae']):.2f}% ± {np.std(metrics['mae']):.2f}%")
    print(f"RMSE: {np.mean(metrics['rmse']):.2f}% ± {np.std(metrics['rmse']):.2f}%")
    print(f"95% CI Coverage: {np.mean(metrics['coverage'])*100:.1f}% ± {np.std(metrics['coverage'])*100:.1f}%")
    
    return metrics

if __name__ == "__main__":
    # Load data from final_model.py
    from final_model import model_df, data
    
    # Only use non-synthetic data for validation
    real_data = model_df[~model_df['is_synthetic']]
    
    print("Running 5-fold cross-validation...")
    metrics = cross_validate(real_data)
