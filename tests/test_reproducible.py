"""Tests for reproducible research utilities."""

import pytest
import yaml
from pathlib import Path
from src.analysis.reproducible import ResearchCompanion

@pytest.fixture
def sample_config():
    """Generate sample configuration."""
    return {
        "model": {
            "draws": 2000,
            "chains": 4,
            "target_accept": 0.95
        },
        "data": {
            "start_year": 1990,
            "end_year": 2023
        },
        "analysis": {
            "cv_folds": 5,
            "prediction_horizon": 2040
        }
    }

def test_report_generation(sample_results, sample_config, tmp_path):
    """Test report generation."""
    companion = ResearchCompanion(
        results=sample_results,
        config=sample_config,
        output_dir=tmp_path
    )
    
    companion.generate_report()
    report_path = tmp_path / "analysis_report.md"
    
    assert report_path.exists()
    with open(report_path) as f:
        content = f.read()
        assert "Gender Disparities Analysis Report" in content
        assert "Model Performance" in content
        assert "2040 Predictions" in content
        
def test_notebook_generation(sample_results, sample_config, tmp_path):
    """Test notebook generation."""
    companion = ResearchCompanion(
        results=sample_results,
        config=sample_config,
        output_dir=tmp_path
    )
    
    companion.generate_notebooks()
    notebook_path = tmp_path / "explore_results.ipynb"
    
    assert notebook_path.exists()
    
def test_complete_generation(sample_results, sample_config, tmp_path):
    """Test complete documentation generation."""
    companion = ResearchCompanion(
        results=sample_results,
        config=sample_config,
        output_dir=tmp_path
    )
    
    companion.generate_all()
    
    assert (tmp_path / "analysis_report.md").exists()
    assert (tmp_path / "explore_results.ipynb").exists()
    assert (tmp_path / "diagnostics").exists()
    assert (tmp_path / "analysis_config.yml").exists()
