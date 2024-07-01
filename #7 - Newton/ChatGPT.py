import numpy as np
import matplotlib.pyplot as plt

# Generate linear data for demonstration
num_arrays = 5
num_points = 100
h_sez_values = np.linspace(0.1, 0.5, num_points)  # Replace with your actual h_sez data

# Generate linear data for exec_results and xerr_results with different coefficients
exec_results = [0.5 * h_sez_values * np.random.rand() + np.random.rand() for _ in range(num_arrays)]
xerr_results = [0.8 * h_sez_values * np.random.rand() + np.random.rand() for _ in range(num_arrays)]

# Define multiple colormaps
cmaps = ['viridis', 'plasma', 'cividis']

# Plot scatter plots with different colormaps
for idx, cmap in enumerate(cmaps):
    plt.scatter(exec_results[idx], xerr_results[idx], c=h_sez_values, cmap=cmap, label=cmap)

    # Add colorbar for each scatter plot
    cbar = plt.colorbar()
    cbar.set_label('h_sez')

# Set labels and title
plt.xlabel('exec_results')
plt.ylabel('xerr_results')
plt.title('Scatter Plot of xerr_results vs. exec_results with Different Colormaps')

# Add legend
plt.legend()

plt.tight_layout()
plt.show()


import numpy as np

# Given array
h_sez = np.array([2.0, 1.5, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])

# Number of additional points between each pair
num_additional_points = 20

# Fill in additional points
new_h_sez = np.hstack([np.linspace(start, stop, num_additional_points, endpoint=False) for start, stop in zip(h_sez[:-1], h_sez[1:])])

# Combine the original array and the additional points
h_sez_combined = np.concatenate([h_sez, new_h_sez, [h_sez[-1]]])

print(h_sez_combined)

