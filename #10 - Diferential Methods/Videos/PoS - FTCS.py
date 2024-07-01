# Pioneer of Success - Python Code for FTCS Heat Equation Solver 

"""
\partial T / \partial t = \alpha \partial^2 T / \partial x^2

0 < x < l;  t > 0

T(x, 0) = 0
T(0, t) = 100
T(l, t) = 0

T(x, t) = ?

n <-- \Delta t - time step
h <-- \Delta x - spac step

Forward Difference:
dT/dt = (T(x, t+n) - T(x, t))/n

Central Difference:
d^2T/dx^2 = (T(x+h, t) - 2T(x, t) + T(x-h, t)) / h^2


\frac{\partial^2 T}{\partial x^2} = 2 * \frac{T(l - \Delta x, t) - T(x, t)}{\Delta x^2}

=> T(x, t+n) = T(x, t) + (n * \alpha / h^2) * [T(x+h, t) - 2T(x, t) + T(x-h, t)]
    T(x, t+n) = T(x, t) + (n * \alpha / h^2) * [2(T(l-h, t) - T(x, t))]
"""

import numpy as np
from scipy import sparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# Number of grids
Nx = 40;    # number of grids along x-axis
Nt = 700;    # number of grids along t-axis

# Define the coordinates
x0 = 0;
xr = 1;

t0 = 0;
tf = 0.2;

dx = (xr - x0) / (Nx - 1);
dt = (tf - t0) / (Nt - 1);

xspan = np.linspace(x0, xr, Nx)
tspan = np.linspace(t0, tf, Nt)

# parameter
alpha = 1;
r = dt*alpha/(dx*dx)

# define a matrix
T = np.zeros((Nx, Nt))

# Boundary conditions
T[0, :] = 100;
T[-1, :] = 0;

# Calculate for interior points
for k in range(0, Nt-1):
    for i in range(1, Nx-1):
        T[i, k+1] = T[i, k] + r*(T[i+1, k] - 2*T[i, k] + T[i-1, k])
        

# Plotting
T1, X = np.meshgrid(tspan, xspan)

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax = fig.gca()

surf = ax.plot_surface(X, T1, T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])

ax.set_xlabel('Space')
ax.set_ylabel('Time')
ax.set_zlabel('T')
ax.view_init(elev=33, azim=36)

plt.tight_layout()
plt.show()