import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility and number of simulations
np.random.seed(42)
N = 1_000_000

# 1. Simulate the random variables
sh_analysis = np.random.exponential(scale=10, size=N)
data_extract = np.maximum(0, np.random.normal(loc=15, scale=3, size=N))
dev_exploration = np.maximum(0, np.random.normal(loc=3, scale=0.5, size=N))
process_integration = np.maximum(0, np.random.normal(loc=50, scale=20, size=N))
model_production = np.maximum(0, np.random.normal(loc=30, scale=5, size=N))
dev_modeling = np.random.poisson(lam=30, size=N)
data_pipeline = np.random.poisson(lam=100, size=N)

# 2. Calculate paths for Base Scenario
track_a = dev_exploration + dev_modeling + model_production
track_b = data_pipeline
parallel_phase = np.maximum(track_a, track_b)
total_time = sh_analysis + data_extract + parallel_phase + process_integration

# 3. Scenario 2: Data Engineer leaves (Pipeline mean = 150)
data_pipeline_new = np.random.poisson(lam=150, size=N)
parallel_phase_new = np.maximum(track_a, data_pipeline_new)
total_time_new = sh_analysis + data_extract + parallel_phase_new + process_integration
delay = total_time_new - total_time

# Set seaborn style for better aesthetics
sns.set_theme(style="whitegrid")

# --- GRAPH 1: Distribution of Total Time ---
plt.figure(figsize=(10, 6))
sns.histplot(total_time, bins=100, kde=True, color='skyblue', stat='density', 
             line_kws={'color': 'darkblue', 'linewidth': 2})
plt.axvline(np.mean(total_time), color='red', linestyle='dashed', linewidth=2, 
            label=f'Mean: {np.mean(total_time):.1f} days')
plt.axvline(np.percentile(total_time, 2.5), color='orange', linestyle='dotted', linewidth=2, 
            label=f'95% CI Lower ({np.percentile(total_time, 2.5):.1f})')
plt.axvline(np.percentile(total_time, 97.5), color='orange', linestyle='dotted', linewidth=2, 
            label=f'95% CI Upper ({np.percentile(total_time, 97.5):.1f})')

plt.title('Distribution of Total Process Time (1,000,000 Simulations)', fontsize=14)
plt.xlabel('Total Time (Days)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

# --- GRAPH 2: Distribution of the Delay ---
plt.figure(figsize=(10, 6))
sns.histplot(delay, bins=50, kde=True, color='salmon', stat='density',
             line_kws={'color': 'darkred', 'linewidth': 2})
plt.axvline(np.mean(delay), color='black', linestyle='dashed', linewidth=2, 
            label=f'Mean Delay: {np.mean(delay):.1f} days')

plt.title('Distribution of Delay (When Pipeline Mean Increases to 150)', fontsize=14)
plt.xlabel('Delay (Days)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

# --- GRAPH 3: Uncertainty Contribution (Variance) ---
variances = {
    'SH Analysis': np.var(sh_analysis),
    'Data Extract': np.var(data_extract),
    'Dev Exploration': np.var(dev_exploration),
    'Dev Modeling': np.var(dev_modeling),
    'Model Production': np.var(model_production),
    'Data Pipeline': np.var(data_pipeline),
    'Process Integration': np.var(process_integration)
}

# Sort variances for better visualization (lowest to highest)
variances_sorted = dict(sorted(variances.items(), key=lambda item: item[1]))

plt.figure(figsize=(10, 6))
sns.barplot(x=list(variances_sorted.values()), y=list(variances_sorted.keys()), palette='viridis')
plt.title('Uncertainty Contribution (Variance) by Process Step', fontsize=14)
plt.xlabel('Variance (Days²)', fontsize=12)
plt.ylabel('Process Step', fontsize=12)
plt.tight_layout()
plt.show()