# Gender Disparities in Clinical Trials Analysis

This repository contains the analysis code for examining gender disparities in clinical trials across COVID-19, Ebola, and HIV studies.

## Setup

1. Create conda environment:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate environment:
   ```bash
   conda activate gender-analysis
   ```

## Usage

Run main analysis:
```bash
python src/run_analysis.py --data-path data/processed/trial_data.csv --results-dir results/
```

Run tests:
```bash
pytest tests/
```

## Project Structure

```
├── data/
│   ├── raw/          # Raw data files
│   └── processed/    # Processed datasets
├── docs/             # Documentation
├── results/
│   ├── figures/      # Generated figures
│   └── tables/       # Generated tables
├── src/              # Source code
│   ├── analysis/     # Core analysis code
│   ├── validation/   # Validation utilities
│   ├── visualization/# Plotting utilities
│   └── utils/        # Helper functions
└── tests/            # Test suite
```

## Documentation

See the `docs/` directory for detailed methodology and results documentation.

## Development

### Code Quality

This project uses several tools to maintain code quality:

1. **Black**: Code formatting
   ```bash
   black src/ tests/
   ```

2. **Flake8**: Style guide enforcement
   ```bash
   flake8 src/ tests/
   ```

3. **MyPy**: Static type checking
   ```bash
   mypy src/ tests/
   ```

4. **Pre-commit**: Automated checks
   ```bash
   pre-commit install
   pre-commit run --all-files
   ```

### Testing

Run tests with coverage:
```bash
pytest --cov=src --cov-report=html
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and quality checks
5. Submit a pull request

