import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

# 1. Bar Chart: FedAvg vs FedRep across all Alphas
alphas = ['0.01', '0.1', '0.5', '1.0', 'IID']
fedavg_accs = [62.50, 80.82, 89.02, 90.83, 91.13]
fedrep_accs = [98.15, 91.18, 81.61, 62.99, 75.97]

x = np.arange(len(alphas))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
rects1 = ax.bar(x - width/2, fedavg_accs, width, label='FedAvg (Baseline)')
rects2 = ax.bar(x + width/2, fedrep_accs, width, label='FedRep (Personalized)')

ax.set_ylabel('Final Test Accuracy (%)', fontsize=12)
ax.set_xlabel('Dirichlet Alpha (Heterogeneity)', fontsize=12)
ax.set_title('Impact of Data Heterogeneity on Federated Learning', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(alphas)
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('results/heterogeneity_comparison.png', bbox_inches='tight')
print("Generated heterogeneity_comparison.png")

# 2. Line Chart: Accuracy vs Rounds for extreme non-IID (Alpha = 0.01)
# Note: These are a subset of your logged rounds to make the chart readable
rounds = list(range(5, 55, 5)) 
fedrep_001_acc = [38.00, 34.78, 53.44, 76.12, 66.14, 93.00, 93.96, 98.44, 98.67, 98.15]
fedavg_001_acc = [24.91, 43.21, 50.71, 54.72, 70.33, 64.97, 55.10, 55.85, 69.42, 62.50]

plt.figure(figsize=(10, 6), dpi=300)
plt.plot(rounds, fedrep_001_acc, label='FedRep (Alpha=0.01)', marker='o', linewidth=2)
plt.plot(rounds, fedavg_001_acc, label='FedAvg (Alpha=0.01)', marker='s', linewidth=2, linestyle='--')

plt.title('Accuracy vs. Communication Rounds (Extreme Non-IID, $\\alpha=0.01$)', fontsize=14)
plt.xlabel('Communication Round', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('results/convergence_alpha_0.01.png', bbox_inches='tight')
print("Generated convergence_alpha_0.01.png")