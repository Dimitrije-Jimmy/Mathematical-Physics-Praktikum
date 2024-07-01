import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color

# Function to calculate results based on parameters
def calculate_result(t, y, k, T0=21.0, T_out=-5.0):
    # Replace this with your actual calculation for heat diffusion
    return T_out + np.exp(-k*t)*(T0 - T_out)

# Number of time points
num_time_points = 100

# Generate time values
time_values = np.linspace(0, 30, num_time_points)

# Generate 10 different starting parameters
k_sez = np.linspace(0.1, 1, 100)

# Generate integration values (replace with your actual values)
integration_values = np.linspace(-10, 30, 100)

# Create a meshgrid for time and integration values
T, Y = np.meshgrid(time_values, k_sez)



# Initialize a figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Create a colormap based on starting parameters
cmap = plt.get_cmap('viridis')
cmap = plt.get_cmap('hot')

# Plot the colormap
heatmap1 = ax1.pcolormesh(T, Y, calculate_result(T, Y, k_sez[:, np.newaxis], 21.0), cmap=cmap)#, norm=color.CenteredNorm(vcenter=0))

# Customize the plot
ax1.set_xlabel(r'Čas [$s$]')
ax1.set_ylabel(r'Vrednosti začetnega parametra - $k$')
ax1.set_title(r'T(t) pri različnih vrednostih parametra $k$ za $T_0 = 21.0\;^{\circ}$C')

# Add a colorbar
cbar1 = plt.colorbar(heatmap1, label='Temperatura - T(t)')

# Plot the colormap
heatmap2 = ax2.pcolormesh(T, Y, calculate_result(T, Y, k_sez[:, np.newaxis], -15.0), cmap=cmap)#, norm=color.CenteredNorm(vcenter=0))

# Customize the plot
ax2.set_xlabel(r'Čas [$s$]')
ax2.set_ylabel(r'Vrednosti začetnega parametra - $k$')
ax2.set_title(r'T(t) pri različnih vrednostih parametra $k$ za $T_0 = -15.0\;^{\circ}$C')

# Add a colorbar
cbar2 = plt.colorbar(heatmap2, label='Temperatura - T(t)')

# Show the plot
plt.show()