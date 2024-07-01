import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import psutil
import os

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
results_list = []

# Perform integration for each step size and store the results
for h in h_sez:
    n = int((b-a)/np.abs(h))
    t = np.linspace( a, b, n )
    numerical_solution = euler(f, x0, t)
    results_list.append((h, numerical_solution))

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(wspace=0.4)

# Plot the mesh plot of the result on the left
axs[0] = fig.add_subplot(121, projection='3d')  # Use projection='3d' for 3D plot
for step_size, numerical_solution in results_list:
    print(step_size)
    T_mesh = np.tile(numerical_solution, (len(np.linspace(0, 1, len(t))), 1))
    T_mesh = T_mesh.T  # Transpose to match the shape

    axs[0].plot_surface(t, np.linspace(0, 1, len(t)),
                        T_mesh,
                        cmap='viridis', edgecolor='k', label=f'Step Size: {step_size}')

axs[0].set_title('Mesh Plot - Numerical Solution')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Step Size')
axs[0].set_zlabel('Temperature (T)')
axs[0].legend()

# Plot the mesh plot of the error on the right
axs[1] = fig.add_subplot(122, projection='3d')  # Use projection='3d' for 3D plot
for step_size, numerical_solution in results_list:
    n = int((b-a)/np.abs(step_size))
    t = np.linspace( a, b, n )
    absolute_error = np.abs(numerical_solution - analyt(t, x0))
    error_mesh = np.tile(absolute_error, (len(np.linspace(0, 1, len(t))), 1))
    error_mesh = error_mesh.T  # Transpose to match the shape

    axs[1].plot_surface(t, np.linspace(0, 1, len(t)),
                        error_mesh,
                        cmap='viridis', edgecolor='k', label=f'Step Size: {step_size}')

axs[1].set_title('Mesh Plot - Absolute Error')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Step Size')
axs[1].set_zlabel('Absolute Error')
axs[1].legend()

plt.show()