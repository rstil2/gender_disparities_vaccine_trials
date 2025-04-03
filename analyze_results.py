#!/usr/bin/env python3
"""
Analyze Results Module

This module provides comprehensive analysis capabilities for the disease prediction model,
including visualization, sensitivity analysis, historical comparison, and documentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import markdown
from pathlib import Path

# Import the final model
import final_model

def plot_predictions(save_path=None):
    """
    Generate plots of model predictions with uncertainty bands for all diseases.
    
    Args:
        save_path (str, optional): Directory to save plots. If None, displays plots instead.
    
    Returns:
        dict: Dictionary of figure objects for each disease.
    """
    # Set seaborn style for better-looking plots
    sns.set_style("whitegrid")
    
    # Create year range for predictions (2023-2040)
    years = np.arange(2023, 2041)
    
    # Dictionary to store figure objects
    figures = {}
    
    # Get predictions for each disease
    diseases = ['COVID-19', 'Ebola', 'HIV']
    for disease in diseases:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get prediction mean and credible intervals
        means, lower_ci, upper_ci = final_model.predict_disease_prevalence(
            disease, years, return_ci=True
        )
        
        # Plot mean prediction
        ax.plot(years, means * 100, 'b-', linewidth=2, label=f"{disease} Prevalence (Mean)")
        
        # Plot uncertainty bands
        ax.fill_between(
            years, 
            lower_ci * 100, 
            upper_ci * 100, 
            alpha=0.3, 
            color='b', 
            label='95% Credible Interval'
        )
        
        # Customize plot
        ax.set_title(f"{disease} Prevalence Projection (2023-2040)", fontsize=16)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Prevalence (%)", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12)
        
        # Add details about model parameters
        model_params = final_model.get_params(disease)
        param_text = f"Model Parameters:\nb0 (intercept): {model_params['b0']:.4f}\nb1 (trend): {model_params['b1']:.4f} ± {model_params['b1_std']:.4f}"
        ax.text(0.02, 0.95, param_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save figures if path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, f"{disease.lower().replace('-', '_')}_projection.png"), 
                        dpi=300, bbox_inches='tight')
        
        figures[disease] = fig
    
    if not save_path:
        plt.show()
        
    return figures

def sensitivity_analysis(save_path=None):
    """
    Perform sensitivity analysis by varying key model parameters and analyzing impact.
    
    Args:
        save_path (str, optional): Directory to save analysis results and plots.
    
    Returns:
        dict: Sensitivity analysis results for each disease.
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Target year for projection comparison
    target_year = 2040
    
    # Parameter variations to test (percentage changes)
    variation_percentages = np.array([-50, -25, -10, 0, 10, 25, 50])
    
    # Dictionary to store analysis results
    sensitivity_results = {}
    
    # Analyze each disease
    diseases = ['COVID-19', 'Ebola', 'HIV']
    for disease in diseases:
        # Get base model parameters
        base_params = final_model.get_params(disease)
        base_b0 = base_params['b0']
        base_b1 = base_params['b1']
        
        # Storage for varied results
        b0_results = []
        b1_results = []
        
        # Vary b0 (intercept)
        for pct in variation_percentages:
            modified_b0 = base_b0 * (1 + pct/100)
            # Use the model to predict with modified b0
            result = final_model.predict_with_params(
                disease, [target_year], modified_b0, base_b1
            )[0] * 100  # Convert to percentage
            b0_results.append(result)
        
        # Vary b1 (trend)
        for pct in variation_percentages:
            # Handle case when b1 is near zero
            if abs(base_b1) < 1e-5:
                modified_b1 = base_b1 + (pct/100) * 0.01  # Arbitrary small change
            else:
                modified_b1 = base_b1 * (1 + pct/100)
            
            # Use the model to predict with modified b1
            result = final_model.predict_with_params(
                disease, [target_year], base_b0, modified_b1
            )[0] * 100  # Convert to percentage
            b1_results.append(result)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot b0 sensitivity
        ax1.plot(variation_percentages, b0_results, 'o-', linewidth=2, color='blue')
        ax1.set_title(f"Sensitivity to Intercept (b0) - {disease}", fontsize=14)
        ax1.set_xlabel("Change in b0 (%)", fontsize=12)
        ax1.set_ylabel(f"Projected {disease} Prevalence in {target_year} (%)", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot b1 sensitivity
        ax2.plot(variation_percentages, b1_results, 'o-', linewidth=2, color='red')
        ax2.set_title(f"Sensitivity to Trend (b1) - {disease}", fontsize=14)
        ax2.set_xlabel("Change in b1 (%)", fontsize=12)
        ax2.set_ylabel(f"Projected {disease} Prevalence in {target_year} (%)", fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, f"{disease.lower().replace('-', '_')}_sensitivity.png"), 
                        dpi=300, bbox_inches='tight')
        
        # Store results
        sensitivity_results[disease] = {
            'b0_variations': variation_percentages.tolist(),
            'b0_results': b0_results,
            'b1_variations': variation_percentages.tolist(),
            'b1_results': b1_results,
            'figure': fig
        }
    
    if not save_path:
        plt.show()
    
    return sensitivity_results

def compare_historical(historical_data=None, save_path=None):
    """
    Compare model predictions with historical data.
    
    Args:
        historical_data (dict, optional): Dictionary of historical data by disease.
            If None, uses example/placeholder data.
        save_path (str, optional): Directory to save comparison plots.
    
    Returns:
        dict: Comparison metrics and figures for each disease.
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # If no historical data provided, use placeholder data
    if historical_data is None:
        # Example historical data format (replace with real data if available)
        historical_data = {
            'COVID-19': {
                'years': [2020, 2021, 2022],
                'prevalence': [0.15, 0.22, 0.20]  # As proportions
            },
            'Ebola': {
                'years': [2014, 2016, 2018, 2020, 2022],
                'prevalence': [0.02, 0.015, 0.012, 0.01, 0.018]
            },
            'HIV': {
                'years': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
                'prevalence': [0.14, 0.137, 0.135, 0.133, 0.13, 0.128, 0.127, 0.125]
            }
        }
    
    # Create figures and compute metrics
    comparison_results = {}
    diseases = ['COVID-19', 'Ebola', 'HIV']
    
    for disease in diseases:
        if disease not in historical_data:
            print(f"Warning: No historical data available for {disease}")
            continue
        
        # Get historical data
        hist_years = np.array(historical_data[disease]['years'])
        hist_prevalence = np.array(historical_data[disease]['prevalence'])
        
        # Get model predictions for the same years
        pred_prevalence, lower_ci, upper_ci = final_model.predict_disease_prevalence(
            disease, hist_years, return_ci=True
        )
        
        # Calculate error metrics
        mae = np.mean(np.abs(pred_prevalence - hist_prevalence))
        rmse = np.sqrt(np.mean((pred_prevalence - hist_prevalence)**2))
        
        # Determine if historical values fall within credible intervals
        within_ci = ((hist_prevalence >= lower_ci) & (hist_prevalence <= upper_ci))
        pct_within_ci = np.mean(within_ci) * 100
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot historical data
        ax.plot(hist_years, hist_prevalence * 100, 'o-', color='black', linewidth=2,
                label='Historical Data')
        
        # Plot model predictions with confidence intervals
        ax.plot(hist_years, pred_prevalence * 100, 's--', color='blue', linewidth=2,
                label='Model Predictions')
        ax.fill_between(
            hist_years, 
            lower_ci * 100, 
            upper_ci * 100, 
            alpha=0.3, 
            color='blue', 
            label='95% Credible Interval'
        )
        
        # Customize plot
        ax.set_title(f"{disease} - Historical vs. Predicted Prevalence", fontsize=16)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Prevalence (%)", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12)
        
        # Add metrics text
        metrics_text = (f"Error Metrics:\n"
                       f"MAE: {mae*100:.2f}%\n"
                       f"RMSE: {rmse*100:.2f}%\n"
                       f"Points within CI: {pct_within_ci:.1f}%")
        
        ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save figure if path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, f"{disease.lower().replace('-', '_')}_comparison.png"), 
                        dpi=300, bbox_inches='tight')
        
        # Store results
        comparison_results[disease] = {
            'mae': mae,
            'rmse': rmse,
            'pct_within_ci': pct_within_ci,
            'figure': fig
        }
    
    if not save_path:
        plt.show()
    
    return comparison_results

def document_model(save_path='model_documentation'):
    """
    Generate comprehensive markdown documentation for the model.
    
    Args:
        save_path (str): Directory to save the documentation.
    
    Returns:
        str: Path to the generated documentation file.
    """
    # Create the docs directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building the documentation
    doc = [
        "# Disease Prevalence Prediction Model Documentation",
        f"\nGenerated on: {timestamp}",
        
        "\n## Overview",
        "This document provides comprehensive documentation for the disease prevalence prediction model.",
        "The model forecasts prevalence for COVID-19, Ebola, and HIV through 2040 using Bayesian methods.",
        
        "\n## Model Architecture",
        "The model uses a Bayesian approach with the following structure:",
        "* Linear trend model: `prevalence = b0 + b1 * (year - reference_year)`",
        "* Bayesian inference for parameter estimation",
        "* Credible intervals to quantify uncertainty",
        
        "\n## Parameters",
    ]
    
    # Add parameters for each disease
    diseases = ['COVID-19', 'Ebola', 'HIV']
    for disease in diseases:
        params = final_model.get_params(disease)
        doc.append(f"\n### {disease}")
        doc.append(f"* Intercept (b0): {params['b0']:.4f}")
        doc.append(f"* Trend (b1): {params['b1']:.4f} ± {params['b1_std']:.4f}")
        
        # Add 2040 projection
        projection_2040 = final_model.predict_disease_prevalence(disease, [2040], return_ci=True)
        doc.append(f"* 2040 Projection: {projection_2040[0][0]*100:.1f}% ({projection_2040[1][0]*100:.1f}% - {projection_2040[2][0]*100:.1f}%)")
    
    # Add methodological details
    doc.extend([
        "\n## Methodology",
        "1. **Data Preparation**: Historical disease prevalence data collected and normalized",
        "2. **Parameter Estimation**: Bayesian inference with appropriate priors",
        "3. **Uncertainty Quantification**: 95% credible intervals via posterior sampling",
        "4. **Validation**: Model validated against historical data where available",
        
        "\n## Assumptions and Limitations",
        "* Assumes linear trends over time (may not capture sudden outbreaks)",
        "* Limite

