import matplotlib.pyplot as plt
import numpy as np

# Data
ratios = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
random = np.array([85.50, 87.40, 88.64, 86.25, 88.01, 88.48, 89.20, 87.26, 82.95, 79.54])
popular = np.array([84.37, 85.74, 86.88, 84.93, 86.27, 86.63, 87.26, 85.73, 82.12, 78.97])
adversarial = np.array([82.32, 82.78, 83.36, 82.14, 83.65, 82.78, 83.00, 82.70, 80.69, 78.32])

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(ratios, random, marker='o', linestyle='-', color='b', label='Random')
plt.plot(ratios, popular, marker='s', linestyle='--', color='g', label='Popular')
plt.plot(ratios, adversarial, marker='^', linestyle='-.', color='r', label='Adversarial')

# Add titles and labels
plt.title('Performance Comparison across Different Ratios', fontsize=16)
plt.xlabel('Ratio', fontsize=14)
plt.ylabel('F1 Score (%)', fontsize=14)

# Customize ticks
plt.xticks(np.arange(0, 1.1, 0.1))  # X-axis ticks with interval of 0.1
plt.ylim(75, 90)  # Y-axis limits

# Customize grid and appearance
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Show plot
plt.show()
