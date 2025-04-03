import pandas as pd
import numpy as np
import pymc as pm
from datetime import datetime
from visualize import create_visualization

# Load and prepare data
print("Loading and preparing data...")

# Example data
data = {
    'COVID-19': {
        'years': [2020, 2021, 2022],
        'rates': [0.34, 0.35, 0.36]
    },
    'Ebola': {
        'years': [2014, 2016, 2018, 2019, 2020],
        'rates': [0.33, 0.34, 0.35, 0.36, 0.37]
    },
    'HIV': {
        'years': [1994, 1998, 2002, 2008, 2012, 2016, 2020],
        'rates': [0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39]
    }
}

# Convert to DataFrame
rows = []
for disease, values in data.items():
    for year, rate in zip(values['years'], values['rates']):
        rows.append({
            'year': year,
            'disease': disease,
            'prevalence': rate * 100,  # Convert to percentage
            'is_synthetic': False
        })

model_df = pd.DataFrame(rows)

print("Calculating participation rates...")

# Generate synthetic data points for gaps
print("Generating synthetic data points...")

def interpolate_years(years, rates):
    all_years = list(range(min(years), max(years) + 1))
    interpolated = np.interp(all_years, years, rates)
    return all_years, interpolated

synthetic_rows = []
for disease in data.keys():
    disease_data = data[disease]
    years = disease_data['years']
    rates = disease_data['rates']
    
    if len(years) > 1:
        all_years, interpolated = interpolate_years(years, rates)
        for year, rate in zip(all_years, interpolated):
            if year not in years:
                synthetic_rows.append({
                    'year': year,
                    'disease': disease,
                    'prevalence': rate * 100,  # Convert to percentage
                    'is_synthetic': True
                })

synthetic_df = pd.DataFrame(synthetic_rows)
model_df = pd.concat([model_df, synthetic_df]).sort_values(['disease', 'year'])

# Print summary statistics
print("\nData Summary:")
print(f"Total data points: {len(model_df)}")
print(f"Real data points: {len(model_df[~model_df['is_synthetic']])}")
print(f"Synthetic data points: {len(model_df[model_df['is_synthetic']])}")

print("\nDetailed Data Summary:\n")
for disease in data.keys():
    disease_data = model_df[model_df['disease'] == disease]
    real_data = disease_data[~disease_data['is_synthetic']]
    synth_data = disease_data[disease_data['is_synthetic']]
    
    print(f"{disease}:")
    print(f"  Real data points: {len(real_data)}")
    print(f"  Synthetic points: {len(synth_data)}")
    print(f"  Year range: {disease_data['year'].min()} - {disease_data['year'].max()}\n")

# Fit model using PyMC
print("Fitting model...")

with pm.Model() as model:
    # Global parameters
    mu_b0 = pm.Normal("mu_b0", mu=0, sigma=1)
    sigma_b0 = pm.HalfNormal("sigma_b0", sigma=1)
    mu_b1 = pm.Normal("mu_b1", mu=0, sigma=0.1)
    sigma_b1 = pm.HalfNormal("sigma_b1", sigma=0.1)
    
    # Disease-specific parameters
    b0_offset = pm.Normal("b0_offset", mu=0, sigma=1, shape=len(data))
    b1_offset = pm.Normal("b1_offset", mu=0, sigma=0.1, shape=len(data))
    
    # Calculate rates
    rates = []
    for i, disease in enumerate(data.keys()):
        disease_data = model_df[model_df['disease'] == disease]
        years = disease_data['year'].values - 2000  # Center years
        
        b0 = mu_b0 + b0_offset[i] * sigma_b0
        b1 = mu_b1 + b1_offset[i] * sigma_b1
        
        logit_p = b0 + b1 * years
        p = pm.math.invlogit(logit_p)
        
        # Likelihood
        pm.Normal("obs_{}".format(disease),
                mu=p,
                sigma=0.1,
                observed=disease_data['prevalence'].values / 100)  # Convert back to proportion
    
    # Sampling
    trace = pm.sample(2000, tune=1000, target_accept=0.9)

# Generate predictions
print("\nGenerating predictions...")

predictions = {}
future_years = np.arange(1990, 2041)
centered_years = future_years - 2000

for i, disease in enumerate(data.keys()):
    print(f"\nGenerating predictions for {disease}")
    
    # Extract posterior distributions
    b0_samples = trace.posterior['mu_b0'].values.flatten() + \
            trace.posterior['b0_offset'].values[:,:,i].flatten() * \
            trace.posterior['sigma_b0'].values.flatten()
    
    b1_samples = trace.posterior['mu_b1'].values.flatten() + \
            trace.posterior['b1_offset'].values[:,:,i].flatten() * \
            trace.posterior['sigma_b1'].values.flatten()
    
    print(f"b0 range: {b0_samples.mean():.3f} ± {b0_samples.std():.3f}")
    print(f"b1 range: {b1_samples.mean():.3f} ± {b1_samples.std():.3f}")
    
    # Generate predictions
    predictions_sample = []
    for b0, b1 in zip(b0_samples, b1_samples):
        logit_p = b0 + b1 * centered_years
        p = 1 / (1 + np.exp(-logit_p))
        predictions_sample.append(p)
    
    predictions_sample = np.array(predictions_sample)
    
    # Store results
    predictions[disease] = {
        'years': future_years,
        'mean': np.mean(predictions_sample, axis=0),
        'lower': np.percentile(predictions_sample, 2.5, axis=0),
        'upper': np.percentile(predictions_sample, 97.5, axis=0)
    }

# Print results
print("\nCalculating credible intervals...")

print("\nResults Summary:")
print("\nBaseline Participation Rates:")

target_year = 2040
print(f"\n{target_year} Projections:")
for disease in data.keys():
    year_idx = np.where(predictions[disease]['years'] == target_year)[0][0]
    mean = predictions[disease]['mean'][year_idx] * 100
    lower = predictions[disease]['lower'][year_idx] * 100
    upper = predictions[disease]['upper'][year_idx] * 100
    print(f"  {disease}: {mean:.1f}% ({lower:.1f}% - {upper:.1f}%)")

# Create visualization
print("\nFormatting predictions for visualization...")
create_visualization(model_df, predictions, np.array(list(data.keys())))

print("\nAnalysis complete. Results have been saved to 'participation_analysis.png'")
