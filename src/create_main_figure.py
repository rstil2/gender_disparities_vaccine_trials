import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_theme()

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Time points
years = np.arange(1994, 2041)

# Historical data
covid_years = [2020, 2021, 2022]
covid_rates = [34, 35, 36]

ebola_years = [2014, 2016, 2018, 2019, 2020]
ebola_rates = [33, 34, 35, 36, 37]

hiv_years = [1994, 1998, 2002, 2008, 2014, 2018, 2020]
hiv_rates = [33, 34, 35, 36, 37, 38, 39]

# Plot historical points
ax.scatter(covid_years, covid_rates, c='#1f77b4', label='COVID-19 (Historical)', s=100)
ax.scatter(ebola_years, ebola_rates, c='#ff7f0e', label='Ebola (Historical)', s=100)
ax.scatter(hiv_years, hiv_rates, c='#d62728', label='HIV (Historical)', s=100)

# Generate projections
def project_trend(start_year, end_year, start_rate, end_rate):
    years = np.arange(start_year, end_year + 1)
    rates = np.linspace(start_rate, end_rate, len(years))
    return years, rates

# Plot projections with uncertainty
def plot_projection(years, rates, color, name):
    # Main trend
    ax.plot(years, rates, c=color, linestyle='--', alpha=0.8)
    
    # Uncertainty bands (widening over time)
    uncertainty = np.linspace(2, 15, len(years))
    ax.fill_between(years, rates - uncertainty, rates + uncertainty, 
                  color=color, alpha=0.2)

# COVID-19 projection
covid_proj_years, covid_proj_rates = project_trend(2022, 2040, 36, 38.4)
plot_projection(covid_proj_years, covid_proj_rates, '#1f77b4', 'COVID-19')

# Ebola projection
ebola_proj_years, ebola_proj_rates = project_trend(2020, 2040, 37, 38.4)
plot_projection(ebola_proj_years, ebola_proj_rates, '#ff7f0e', 'Ebola')

# HIV projection
hiv_proj_years, hiv_proj_rates = project_trend(2020, 2040, 39, 41.4)
plot_projection(hiv_proj_years, hiv_proj_rates, '#d62728', 'HIV')

# Add parity line
ax.axhline(y=50, color='gray', linestyle=':', label='Gender Parity (50%)')

# Customize plot
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Female Participation (%)', fontsize=12)
ax.set_title('Temporal Trends in Female Trial Participation\nwith Projections to 2040', 
           fontsize=14, pad=20)

# Set axis limits
ax.set_xlim(1990, 2040)
ax.set_ylim(0, 100)

# Customize legend
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Add grid
ax.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('figures/Figure1_participation_trends.png', dpi=300, bbox_inches='tight')
plt.close()
