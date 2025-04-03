import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('gender_representation_analysis.csv')

# Extract data for plotting
years = df['year'].astype(str)
diseases = df['disease']
trial_percentages = df['trial_female_percentage']
population_percentages = df['population_female_percentage']

# Create x-axis labels combining year and disease
x_labels = [f"{year}\n({disease})" for year, disease in zip(years, diseases)]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set width of bars
bar_width = 0.35
x = np.arange(len(years))

# Create bars
trial_bars = ax.bar(x - bar_width/2, trial_percentages, bar_width, 
                   label='Trial Female %', color='#5DA5DA', edgecolor='black')
pop_bars = ax.bar(x + bar_width/2, population_percentages, bar_width,
                 label='Population Female %', color='#FAA43A', edgecolor='black')

# Add labels, title and legend
ax.set_xlabel('Year (Disease)', fontsize=12, fontweight='bold')
ax.set_ylabel('Female Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Gender Representation in Clinical Trials vs. US Population', 
            fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend(loc='upper left', frameon=True)

# Add grid for readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Set y-axis to start at 0 and have some headroom
ax.set_ylim(0, max(population_percentages.max(), trial_percentages.max()) * 1.1)

# Add value labels on top of bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontweight='bold')

add_labels(trial_bars)
add_labels(pop_bars)

# Add a note about the gap
for i, gap in enumerate(df['representation_gap']):
    ax.annotate(f'Gap: {gap:.1f}%',
               xy=(i, min(trial_percentages[i], population_percentages[i]) / 2),
               ha='center', va='center',
               color='#B22222', fontweight='bold')

# Adjust layout and save
plt.tight_layout()
plt.savefig('gender_representation.png', dpi=300)
print("Plot saved as 'gender_representation.png'")

