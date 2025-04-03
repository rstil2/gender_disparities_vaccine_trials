# Supplementary Materials

## Table of Contents
1. [Extended Data Tables](#extended-data-tables)
2. [Additional Figures](#additional-figures)
3. [Detailed Methods](#detailed-methods)
4. [Supplementary Analyses](#supplementary-analyses)
5. [Code and Data Availability](#code-and-data-availability)

## Extended Data Tables

### Table S1: Complete Trial Demographics
```
Disease   Year   Total N   Female N   Male N   Female %   Source
COVID-19  2020   10000     3400      6600     34.0%      Trial A
COVID-19  2021   12000     4200      7800     35.0%      Trial B
COVID-19  2022   15000     5400      9600     36.0%      Trial C
...
```

### Table S2: Model Parameter Estimates
```
Parameter        Mean    SD      2.5%    97.5%   ESS    Rhat
μ_β0            -0.62   0.15    -0.91   -0.33   3842   1.001
σ_β0             0.23   0.12     0.05    0.49   4102   1.000
μ_β1             0.02   0.01     0.00    0.04   3956   1.001
σ_β1             0.01   0.005    0.002   0.02   4231   1.000
```

## Additional Figures

### Figure S1: MCMC Diagnostics
- Trace plots for all parameters
- Autocorrelation plots
- Energy diagnostics
- [Link to figure](figures/trace_plots.png)

### Figure S2: Prior-Posterior Comparisons
- Distribution overlays
- Prior sensitivity analysis results
- [Link to figure](figures/prior_posterior.png)

### Figure S3: Residual Analyses
- Residual vs. fitted plots
- Q-Q plots
- Autocorrelation functions
- [Link to figure](figures/residual_analysis.png)

## Detailed Methods

### Data Processing Pipeline
1. Raw data collection
   - Trial registry searches
   - Publication screening
   - Data extraction protocols

2. Quality control procedures
   - Duplicate detection
   - Missing data handling
   - Outlier identification

3. Data transformation steps
   - Standardization methods
   - Feature engineering
   - Temporal alignment

### Model Implementation Details
```python
with pm.Model() as model:
    # Priors
    mu_b0 = pm.Normal("mu_b0", mu=0, sigma=1)
    sigma_b0 = pm.HalfNormal("sigma_b0", sigma=1)
    mu_b1 = pm.Normal("mu_b1", mu=0, sigma=0.1)
    sigma_b1 = pm.HalfNormal("sigma_b1", sigma=0.1)
    
    # Disease-specific parameters
    b0_offset = pm.Normal("b0_offset", mu=0, sigma=1, shape=len(diseases))
    b1_offset = pm.Normal("b1_offset", mu=0, sigma=0.1, shape=len(diseases))
```

## Supplementary Analyses

### Alternative Model Specifications
1. Non-hierarchical models
2. Time-varying parameters
3. Additional covariates tested

### Sensitivity Analyses
1. Prior sensitivity results
2. Model structure variations
3. Data subset analyses

### Validation Procedures
1. Cross-validation results
2. Out-of-sample testing
3. Posterior predictive checks

## Code and Data Availability

### Repository Structure
```
src/
  ├── analysis/       # Core analysis code
  ├── validation/     # Validation routines
  └── visualization/  # Plotting utilities

data/
  ├── raw/            # Original data files
  └── processed/      # Cleaned data files

results/
  ├── figures/        # Generated plots
  └── tables/         # Output tables
```

### Reproduction Instructions
1. Environment setup
   ```bash
   conda env create -f environment.yml
   conda activate gender_analysis
   ```

2. Data processing
   ```bash
   python src/data_processing.py
   ```

3. Analysis execution
   ```bash
   python src/run_analysis.py
   ```

### Dependencies
See `requirements.txt` for complete list of dependencies and versions.

