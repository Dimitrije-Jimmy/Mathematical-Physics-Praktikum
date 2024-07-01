import numpy as np
import matplotlib.pyplot as plt
import time
import psutil

import matplotlib.colors
from matplotlib.collections import LineCollection

from diffeq import *


def f( x, t, k=0.1, T_out=-5.0 ):
    return -k*(x - T_out)

def analyt( t, x0, k=0.1, T_out=-5.0 ):
    return  T_out + np.exp(-k*t)*(x0 - T_out)


a, b = ( 0.0, 31.0 )

x0 = 21.0
t = np.linspace( a, b, 10 )
x_analyt = analyt(t, x0)

h_sez = [2.0, 1.0, 0.5, 0.1]#, 0.001, 1e-4]


# Initialize lists to store results for each step size
time_values = []
integration_values = []

# Perform integration for each step size and store the results
for h in h_sez:
    n = int((b-a)/np.abs(h))
    t = np.linspace( a, b, n )
    numerical_solution = euler(f, x0, t)
    integration_values.append(numerical_solution)
    time_values.append(t)


"""
# Create a meshgrid for time and integration values
T, Y = np.meshgrid(time_values, integration_values[:, 1])

# Generate 10 different starting parameters
starting_parameters = np.linspace(1, 10, 10)

# Initialize a figure and axis
fig, ax = plt.subplots()

# Create a colormap based on starting parameters
cmap = plt.get_cmap('viridis')

# Plot the colormap
heatmap = ax.pcolormesh(T, Y, integration_values, cmap=cmap)


# Add a colorbar
cbar = plt.colorbar(heatmap, label='Starting Parameter')
"""

# Initialize a figure and axis
fig, ax = plt.subplots()

segments = [np.column_stack((x, y)) for x, y in zip(time_values, integration_values)]
#print(segments)
col = LineCollection(segments, cmap="plasma")#, norm=matplotlib.colors.LogNorm(vmin=p_start, vmax=p_end))
col.set_array(h_sez)
ax.add_collection(col, autolim=True)
plt.colorbar(mappable=col, label="Velikost koraka h")

# Customize the plot
ax.set_xlabel('Time (t)')
ax.set_ylabel('Integration Values (y)')
ax.set_title('Colormap of Heat Diffusion with Different Starting Parameters')


# Show the plot
plt.show()
