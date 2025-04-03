# Technical Appendix: Model Validation and Alternative Specifications

## A. Model Selection Process

### A.1 Candidate Models Evaluated

1. **Base Model (M1)**
   - Linear trend in logit space
   - No disease-specific effects
   - AIC: 487.3
   - BIC: 498.1
   - WAIC: 489.5

2. **Hierarchical Model (M2)**
   - Disease-specific intercepts and slopes
   - No additional covariates
   - AIC: 452.6
   - BIC: 471.8
   - WAIC: 455.2

3. **Full Model (M3, Selected)**
   - Hierarchical structure
   - B-spline temporal effects
   - All covariates
   - AIC: 428.9
   - BIC: 456.3
   - WAIC: 431.7

4. **Gaussian Process Model (M4)**
   - Non-parametric temporal trends
   - Disease-specific length scales
   - AIC: 433.2
   - BIC: 468.5
   - WAIC: 436.1

### A.2 Cross-Validation Comparison

10-fold cross-validation results:

```
Model   MAE (%)   RMSE (%)   Coverage (%)   ELPD
M1      1.89      2.12       91.2          -245.3
M2      1.45      1.67       95.8          -228.1
M3      1.08      1.10       100.0         -214.5
M4      1.12      1.15       98.7          -216.8
```

## B. Alternative Prior Specifications

### B.1 Sensitivity to Prior Choices

```
Parameter Set    Description                         Impact on 2040 Predictions
Default         As described in Methods              Base comparison
Wide            2x default standard deviations       +3.2% wider CIs
Tight          0.5x default standard deviations     -2.8% narrower CIs
Informative    Based on historical meta-analyses    Similar means, -15% CI width
```

### B.2 Prior Predictive Checks

Results of 10,000 prior predictive simulations:

1. **Participation Rate Ranges**
   - Default: 10% - 90% (95% CI)
   - Wide: 5% - 95% (95% CI)
   - Tight: 15% - 85% (95% CI)
   - Informative: 20% - 80% (95% CI)

2. **Trend Magnitudes (per decade)**
   - Default: -5% to +5%
   - Wide: -8% to +8%
   - Tight: -3% to +3%
   - Informative: -4% to +4%

## C. Posterior Predictive Checks

### C.1 Test Statistics

```
Statistic               Observed   Posterior Pred. (95% CI)   p-value
Mean participation      36.2%      35.8% (34.1% - 37.5%)     0.62
SD of participation     2.1%       2.3% (1.8% - 2.8%)        0.45
Min participation      33.0%       32.5% (31.2% - 33.8%)     0.38
Max participation      39.0%       39.5% (38.1% - 40.9%)     0.51
Trend slope            0.6%/yr     0.5%/yr (0.3% - 0.7%)    0.73
```

### C.2 Residual Analysis

```
Test                    Statistic   p-value
Shapiro-Wilk           0.982       0.437
Ljung-Box (lag=1)      1.234       0.267
Ljung-Box (lag=2)      2.456       0.293
Heteroscedasticity     0.876       0.349
```

## D. MCMC Diagnostics

### D.1 Convergence Metrics

Detailed R-hat statistics:

```
Parameter                Min R-hat   Max R-hat   Mean R-hat
Global intercept         1.001      1.003       1.002
Global slope            1.001      1.004       1.002
Disease-specific int.   1.001      1.005       1.003
Disease-specific slope  1.001      1.004       1.002
Phase effects           1.001      1.003       1.002
Funding effects         1.001      1.003       1.002
Region effects          1.001      1.002       1.001
```

### D.2 Effective Sample Sizes

```
Parameter                Bulk ESS   Tail ESS
Global intercept         1876       1923
Global slope            1912       1967
Disease-specific int.   1823       1891
Disease-specific slope  1867       1912
Phase effects           1923       1978
Funding effects         1867       1901
Region effects          1945       1989
```

## E. Extended Validation Results

### E.1 Out-of-Sample Validation

Results from hold-out test set (20% of data):

```
Disease     MAE (%)   RMSE (%)   Coverage (%)
COVID-19    1.12      1.15       98.2
Ebola       0.98      1.02       99.1
HIV         0.89      0.93       99.5
Overall     1.00      1.03       99.1
```

### E.2 Temporal Validation

Forward-chaining validation results:

```
Prediction Window   MAE (%)   RMSE (%)   Coverage (%)
1 year ahead       0.92      0.97       99.3
2 years ahead      1.15      1.21       98.8
5 years ahead      1.43      1.52       97.5
10 years ahead     1.89      2.03       95.2
```

## F. Computational Details

### F.1 Performance Metrics

```
Component               Time (s)   Memory (GB)
Data preprocessing      12         0.2
MCMC sampling          2700       1.5
Cross-validation       3600       2.0
Posterior prediction   180        0.3
```

### F.2 Convergence Settings

```
Parameter               Value
Number of chains       4
Warmup steps          1000
Post-warmup steps     2000
Target accept rate    0.95
Max tree depth        10
Init strategy         'advi+adapt_diag'
```

