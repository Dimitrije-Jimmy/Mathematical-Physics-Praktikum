import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Function to calculate results based on parameters
def calculate_result(x, y):
    return x**2+y  # Replace this with your actual calculation

# Number of points for the meshgrid
num_points = 100

# Generate 10 different starting parameters
starting_parameters = np.linspace(-5, 5, 10)

# Create a meshgrid
x = np.linspace(0, 10, num_points)
y = np.linspace(0, 10, num_points)
X, Y = np.meshgrid(x, y)

# Initialize a figure and axis
fig, ax = plt.subplots()

# Create a continuous colormap
cmap = plt.get_cmap('viridis')

# Plot the mesh surface as a filled contour plot
contour = ax.contourf(X, Y, calculate_result(X, Y), cmap=cmap)

# Plot the 10 starting parameters
#for param in starting_parameters:
#    ax.plot(param, param, marker='o', color='red', markersize=8)

# Customize the plot
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Continuous Colormap with 10 Starting Parameters')

# Add a colorbar
cbar = plt.colorbar(contour)
cbar.set_label('Z-axis')

# Show the plot
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Function to calculate results based on parameters
def calculate_result(t, y, starting_parameter):
    # Replace this with your actual calculation for heat diffusion
    return -5.0 + np.exp(-starting_parameter*0.1*t)*(21.0 - -5.0)

# Number of time points
num_time_points = 100

# Generate time values
time_values = np.linspace(0, 10, num_time_points)

# Generate integration values (replace with your actual values)
integration_values = np.linspace(-5, 5, 10)

# Create a meshgrid for time and integration values
T, Y = np.meshgrid(time_values, integration_values)

# Generate 10 different starting parameters
starting_parameters = np.linspace(1, 100, 10)

# Initialize a figure and axis
fig, ax = plt.subplots()

# Create a colormap based on starting parameters
cmap = plt.get_cmap('viridis')

# Plot the colormap
heatmap = ax.pcolormesh(T, Y, calculate_result(T, Y, starting_parameters[:, np.newaxis]), cmap=cmap)

# Customize the plot
ax.set_xlabel('Time (t)')
ax.set_ylabel('Integration Values (y)')
ax.set_title('Colormap of Heat Diffusion with Different Starting Parameters')

# Add a colorbar
cbar = plt.colorbar(heatmap, label='Starting Parameter')

# Show the plot
plt.show()
