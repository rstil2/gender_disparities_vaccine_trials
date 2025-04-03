# Technical Appendix

## 1. Detailed Model Specification

### 1.1 Mathematical Formulation
The hierarchical model is specified as follows:

For each disease i and year t:
```
logit(p_it) = β0_i + β1_i * (year_t - 2000)

where:
β0_i = μ_β0 + δ0_i * σ_β0
β1_i = μ_β1 + δ1_i * σ_β1

δ0_i ~ Normal(0, 1)
δ1_i ~ Normal(0, 1)
```

### 1.2 Prior Justification
- The normal prior on μ_β0 centers the baseline log-odds around 0 (50% participation)
- The smaller prior scale on μ_β1 reflects our belief in gradual rather than abrupt changes
- Half-normal priors on σ parameters ensure positive variance while being minimally informative

### 1.3 Convergence Diagnostics
- Gelman-Rubin statistics (R̂) calculated for all parameters
- Effective sample sizes (ESS) monitored
- Trace plots examined for mixing and stationarity
- Energy plots checked for sampling efficiency

## 2. Data Processing Details

### 2.1 Data Cleaning Procedures
1. Removal of duplicate entries
2. Handling of missing data
3. Validation of date formats
4. Quality checks for participation rates

### 2.2 Interpolation Method
Linear interpolation was chosen based on:
- Simplicity and interpretability
- Reasonable fit to observed patterns
- Conservative estimates for missing values

### 2.3 Data Quality Metrics
- Completeness of records
- Consistency of reporting
- Temporal coverage
- Source reliability

## 3. Extended Sensitivity Analyses

### 3.1 Prior Sensitivity
Alternative priors tested:
- Wider/narrower normal distributions
- Student's t distributions
- Cauchy distributions

### 3.2 Model Variants
Additional models evaluated:
1. Non-hierarchical individual disease models
2. Shared slope model
3. Non-linear time trends
4. Additional covariates

### 3.3 Cross-validation Results
- Leave-one-out cross-validation
- K-fold cross-validation
- Posterior predictive checks

## 4. Computational Details

### 4.1 Software Versions
- Python 3.10
- PyMC 5.x
- NumPy 1.24.x
- Pandas 2.1.x
- Matplotlib 3.8.x
- Seaborn 0.13.x

### 4.2 Computational Resources
- Hardware specifications
- Runtime measurements
- Memory usage
- Parallel processing details

### 4.3 Reproducibility Information
- Random seed settings
- Environment configuration
- Data preprocessing pipeline
- Analysis workflow

## 5. Additional Visualizations

### 5.1 Diagnostic Plots
- MCMC trace plots
- Posterior distributions
- Residual analyses
- Prediction error distributions

### 5.2 Supplementary Figures
- Individual disease trends
- Prior-posterior comparisons
- Model validation plots
- Sensitivity analysis results

## 6. Extended Results

### 6.1 Parameter Estimates
Detailed posterior summaries for all model parameters:
- Mean, median, standard deviation
- 95% credible intervals
- Mode and skewness
- Autocorrelation statistics

### 6.2 Model Comparisons
Comparison metrics across model variants:
- WAIC scores
- LOO-CV results
- DIC values
- Bayes factors where applicable

### 6.3 Predictions
Extended prediction tables:
- 5-year intervals to 2040
- Multiple probability thresholds
- Disease-specific projections
- Joint probability estimates

