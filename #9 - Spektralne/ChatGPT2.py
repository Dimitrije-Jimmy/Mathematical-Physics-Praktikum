import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Set up parameters
a, b, n = 0., 1., 100
L = 1.
D = 0.05
dt = 0.01
n_steps = 100
sigma = L/4.

# Create spatial grid
t= np.linspace(0, 20, n)
x = np.linspace(a, b, n)

dx = x[1] - x[0]
dt = t[1] - t[0]

# Initialize temperature matrix
temperature_values = np.zeros((n, n_steps))

# Set initial condition: sin(pi*x)
temperature_values[:, 0] = np.sin(np.pi * x)
temperature_values[:, 0] = np.exp(-((x - L/2.)**2.)/(sigma**2.))

# Construct matrices A and B
A = np.zeros((n, n))
B = np.zeros((n, n))

np.fill_diagonal(A, 4)
np.fill_diagonal(A[1:], 1)
np.fill_diagonal(A[:, 1:], 1)
A[0, 1] = 1
A[-1, -2] = 1

np.fill_diagonal(B, -2)
np.fill_diagonal(B[1:], 1)
np.fill_diagonal(B[:, 1:], 1)
B[0, 1] = 1
B[-1, -2] = 1
#B = 6 * D / dx**2 * B
B = 6 * D * dt / (b-a)**2 * B
"""
# Time-stepping loop
for step in range(1, n_steps):
    ab = [B[-1, :], A.diagonal(), B[1, :]]
    rhs = np.dot(A + dt/2 * B, temperature_values[:, step - 1])
    #rhs = np.dot(1 + dt*np.linalg.inv(A)*B, temperature_values[:, step - 1])
    temperature_values[:, step] = solve_banded((1, 1), ab, rhs)
"""

# Explicit finite differences time-stepping loop
for step in range(1, n_steps):
    for i in range(1, n - 1):
        temperature_values[i, step] = temperature_values[i, step - 1] + 6*D  / ((b - a) / (n - 1))**2 * (
            temperature_values[i + 1, step - 1] - 2 * temperature_values[i, step - 1] + temperature_values[i - 1, step - 1]
        )
# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for space and time
X, T = np.meshgrid(x, t)

# Plot the surface
surf = ax.plot_surface(X, T, temperature_values.T, cmap='jet', edgecolor='k')

# Customize the plot
ax.set_xlabel('Space (x)')
ax.set_ylabel('Time (t)')
ax.set_zlabel('Temperature (T)')
ax.set_title('3D Plot of Temperature Evolution')

# Add colorbar
fig.colorbar(surf, ax=ax, label='Temperature')

# Show the plot
plt.show()
