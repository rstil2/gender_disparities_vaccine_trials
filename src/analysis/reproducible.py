"""Reproducible research utilities and documentation generators."""

import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from .results import Results

class ResearchCompanion:
    """Documentation and reproducibility companion for analysis."""
    
    def __init__(
        self,
        results: Results,
        config: Dict[str, Any],
        output_dir: Path
    ):
        self.results = results
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def generate_report(self) -> None:
        """Generate comprehensive analysis report."""
        report = []
        
        # Header
        report.extend([
            "# Gender Disparities Analysis Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Analysis Configuration",
            "```yaml"
        ])
        
        # Configuration details
        report.extend([
            yaml.dump(self.config, default_flow_style=False),
            "```",
            "",
            "## Results Summary",
            ""
        ])
        
        # Performance metrics
        report.extend([
            "### Model Performance",
            "",
            "| Metric | Value |",
            "|--------|--------|"
        ])
        
        for metric, value in self.results.metrics.items():
            report.append(f"| {metric.upper()} | {value:.3f} |")
        
        # Predictions
        report.extend([
            "",
            "### 2040 Predictions",
            "",
            "| Disease | Median | 95% CI |",
            "|---------|---------|---------|"
        ])
        
        for disease, pred in self.results.predictions.items():
            report.append(
                f"| {disease} | {pred['median']:.1%} | ({pred['lower']:.1%} - {pred['upper']:.1%}) |"
            )
        
        # Save report
        report_path = self.output_dir / "analysis_report.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report))
            
    def generate_notebooks(self) -> None:
        """Generate Jupyter notebooks for result exploration."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# Gender Disparities Analysis Results\n",
                        "This notebook provides interactive exploration of analysis results."
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "import numpy as np\n",
                        "import pandas as pd\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        "from pathlib import Path"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Load Results"]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# Load predictions\n",
                        "predictions = pd.read_csv('tables/predictions.csv')\n",
                        "metrics = pd.read_csv('tables/metrics.csv')"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        notebook_path = self.output_dir / "explore_results.ipynb"
        with open(notebook_path, "w") as f:
            json.dump(notebook, f, indent=2)
            
    def generate_diagnostics(self) -> None:
        """Generate diagnostic visualizations and summaries."""
        diag_dir = self.output_dir / "diagnostics"
        diag_dir.mkdir(exist_ok=True)
        
        # Trace plots
        self.results.trace.plot()
        plt.savefig(diag_dir / "trace_plots.png")
        plt.close()
        
        # Convergence statistics
        summary = az.summary(self.results.trace)
        summary.to_csv(diag_dir / "convergence_stats.csv")
        
    def generate_all(self) -> None:
        """Generate all documentation and companion materials."""
        self.generate_report()
        self.generate_notebooks()
        self.generate_diagnostics()
        
        # Copy configuration
        config_path = self.output_dir / "analysis_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        print(f"Research companion materials generated in: {self.output_dir}")
