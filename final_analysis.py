import numpy as np
import pandas as pd
import xarray as xr
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set random seed for reproducibility
np.random.seed(42)

# Load and prepare data
print("Loading and preparing data...")
df = pd.read_csv('trial_data.csv')
df['year_centered'] = df['year'] - 2000
diseases = sorted(df['disease'].unique())

# Create participation rates DataFrame
print("Calculating participation rates...")
rates_data = []
for _, row in df.iterrows():
    rates_data.append({
        'year': row['year'],
        'disease': row['disease'],
        'rate': row['females'] / row['participants'],
        'participants': row['participants'],
        'females': row['females']
    })
rates_df = pd.DataFrame(rates_data)

# Generate synthetic points
print("Generating synthetic data points...")
synthetic_data = []
for disease in diseases:
    disease_data = rates_df[rates_df['disease'] == disease].sort_values('year')
    years = disease_data['year'].values
    rates = disease_data['rate'].values
    
    for i in range(len(years) - 1):
        if years[i + 1] - years[i] > 1:
            # Create one synthetic point between years
            mid_year = (years[i] + years[i + 1]) / 2
            mid_rate = (rates[i] + rates[i + 1]) / 2
            avg_participants = int(disease_data['participants'].mean())
            
            synthetic_data.append({
                'year': mid_year,
                'disease': disease,
                'rate': mid_rate,
                'participants': avg_participants,
                'females': int(mid_rate * avg_participants),
                'is_synthetic': True
            })

# Add is_synthetic column to original data
rates_df['is_synthetic'] = False

# Combine real and synthetic data
if synthetic_data:
    synthetic_df = pd.DataFrame(synthetic_data)
    model_df = pd.concat([rates_df, synthetic_df], ignore_index=True)
else:
    model_df = rates_df.copy()

# Add year_centered to model_df
model_df['year_centered'] = model_df['year'] - 2000
model_df = model_df.sort_values(['disease', 'year'])

print("\nData Summary:")
print(f"Total data points: {len(model_df)}")
print(f"Real data points: {len(rates_df)}")
print(f"Synthetic data points: {len(synthetic_data)}")

# Print detailed data summary
print("\nDetailed Data Summary:")
for disease in diseases:
    disease_data = model_df[model_df['disease'] == disease]
    real_data = disease_data[~disease_data['is_synthetic']]
    synth_data = disease_data[disease_data['is_synthetic']]
    print(f"\n{disease}:")
    print(f"  Real data points: {len(real_data)}")
    print(f"  Synthetic points: {len(synth_data)}")
    print(f"  Year range: {disease_data['year'].min():.0f} - {disease_data['year'].max():.0f}")

# Define the PyMC model
def create_model(data):
    coords = {"disease": diseases}
    with pm.Model(coords=coords) as model:
        # More informative priors for baseline rates
        mu_b0 = pm.Normal('mu_b0', mu=0, sigma=0.5)
        sigma_b0 = pm.HalfNormal('sigma_b0', sigma=0.3)
        
        b0_offset = pm.Normal('b0_offset', mu=0, sigma=1, dims='disease')
        b0 = pm.Deterministic('b0', mu_b0 + b0_offset * sigma_b0, dims='disease')
        
        # More informative priors for growth rates
        mu_b1 = pm.Normal('mu_b1', mu=0, sigma=0.05)
        sigma_b1 = pm.HalfNormal('sigma_b1', sigma=0.03)
        
        b1_offset = pm.Normal('b1_offset', mu=0, sigma=1, dims='disease')
        b1 = pm.Deterministic('b1', mu_b1 + b1_offset * sigma_b1, dims='disease')
        
        # Model with saturation
        disease_idx = pd.Categorical(data['disease'], categories=diseases).codes
        logit_p = b0[disease_idx] + b1[disease_idx] * data['year_centered']
        p = pm.Deterministic('p', pm.math.invlogit(pm.math.clip(logit_p, -4, 4)))
        
        # Likelihood
        y = pm.Binomial('y', n=data['participants'], p=p, observed=data['females'])
    
    return model

# Create and sample from the model
print("\nFitting model...")
with create_model(model_df):
    trace = pm.sample(2000, tune=1000, target_accept=0.99, return_inferencedata=True)

def generate_predictions(trace, years=np.arange(2020, 2041)):
    years_centered = years - 2000
    
    # Extract posterior samples and keep original dimensions
    b0_samples = trace.posterior['b0']  # shape: (chain, draw, disease)
    b1_samples = trace.posterior['b1']
    
    # Generate predictions maintaining dimensions
    n_chains = b0_samples.sizes['chain']
    n_draws = b0_samples.sizes['draw']
    
    predictions = xr.DataArray(
        np.zeros((n_chains, n_draws, len(diseases), len(years))),
        dims=['chain', 'draw', 'disease', 'year'],
        coords={
            'chain': range(n_chains),
            'draw': range(n_draws),
            'disease': diseases,
            'year': years
        }
    )
    
    for d, disease in enumerate(diseases):
        disease_b0 = b0_samples.sel(disease=disease)
        disease_b1 = b1_samples.sel(disease=disease)
        
        for y, year in enumerate(years_centered):
            logit_p = disease_b0 + disease_b1 * year
            predictions.loc[:, :, disease, years[y]] = 1 / (1 + np.exp(-np.clip(logit_p, -4, 4)))
    
    return predictions

# Generate predictions
print("\nGenerating predictions...")
predictions = generate_predictions(trace)

# Calculate HDI for predictions
print("\nCalculating HDIs...")
hdi_data = {}
for disease in diseases:
    disease_predictions = predictions.sel(disease=disease, year=2040)
    hdi = az.hdi(disease_predictions)
    hdi_array = hdi.values.flatten()  # Convert to flat array
    hdi_data[disease] = {
        'lower': float(hdi_array[0]),  # First value is lower bound
        'upper': float(hdi_array[1])   # Second value is upper bound
    }

# Print results
print("\nResults Summary:")
print("\nBaseline Participation Rates:")
for disease in diseases:
    b0_mean = float(trace.posterior['b0'].sel(disease=disease).mean())
    prob = 1 / (1 + np.exp(-b0_mean))
    print(f"  {disease}: {prob:.1%}")

print("\n2040 Projections:")
for disease in diseases:
    median = float(predictions.sel(disease=disease, year=2040).mean(['chain', 'draw']))
    lower = hdi_data[disease]['lower']
    upper = hdi_data[disease]['upper']
    print(f"  {disease}: {median:.1%} ({lower:.1%} - {upper:.1%})")

# Create visualization
print("\nCreating visualization...")
plt.style.use('seaborn')
colors = plt.cm.Set2(np.linspace(0, 1, len(diseases)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 2])
fig.suptitle('Gender Participation in Clinical Trials', fontsize=16, y=0.95)

# Forest plot of baseline rates
az.plot_forest(trace, var_names=['b0'], combined=True, ax=ax1)
ax1.set_title('Baseline Participation Rates', fontsize=14)

# Plot trajectories and predictions
for disease, color in zip(diseases, colors):
    # Plot real data
    real_data = model_df[(model_df['disease'] == disease) & (~model_df['is_synthetic'])]
    ax2.scatter(real_data['year'], real_data['rate'],
            label=f'{disease} (Real)', color=color, marker='o', s=100, zorder=5)
    
    # Plot synthetic data points
    synth_data = model_df[(model_df['disease'] == disease) & (model_df['is_synthetic'])]
    if len(synth_data) > 0:
        ax2.scatter(synth_data['year'], synth_data['rate'],
                label=f'{disease} (Synthetic)', color=color, marker='x', alpha=0.5, s=50, zorder=4)
    
    # Plot median predictions
    disease_pred = predictions.sel(disease=disease)
    median = disease_pred.mean(['chain', 'draw'])
    ax2.plot(predictions.year, median, color=color, zorder=3)
    
    # Plot HDI
    lower = disease_pred.quantile(0.025, dim=['chain', 'draw'])
    upper = disease_pred.quantile(0.975, dim=['chain', 'draw'])
    ax2.fill_between(predictions.year, lower, upper, color=color, alpha=0.2, zorder=2)

# Add reference line for gender parity
ax2.axhline(0.5, ls='--', color='gray', label='Gender Parity', zorder=1)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Female Participation Rate', fontsize=12)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('participation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis complete. Results have been saved to 'participation_analysis.png'")

