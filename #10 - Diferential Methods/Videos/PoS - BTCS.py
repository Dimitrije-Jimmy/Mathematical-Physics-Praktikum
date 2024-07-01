# Pioneer of Success - Python code BTCS scheme to solve Unsteady Heat Conduction

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
dT/dt = (T(x, t+n) - T(x, t)) / n

Central Difference:
d^2T/dx^2 = (T(x+h, t+n) - 2T(x, t+n) + T(x-h, t+n)) / h^2


\frac{\partial^2 T}{\partial x^2} = 2 * \frac{T(l - \Delta x, t) - T(x, t)}{\Delta x^2}

=> T(x, t+n) = T(x, t) + (n * \alpha / h^2) * [T(x+h, t+n) - 2T(x, t+n) + T(x-h, t+n)]
    T(x, t+n) = T(x, t) + (n * \alpha / h^2) * [2(T(l-h, t+n) - T(x, t+n))]
"""


'''
Backward Time Central Space (BTCS) method to solve 1D Heat Conduction equation:
    T-t = alpha * T_xx
    
with Dirichlet boundary conditions T(x0, t) = 0, T(xL, t) = 0
and initial condition u(x, 0) = 4*x - 4*x**2
'''

import numpy as np
from scipy import sparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# Set number of grid points
M = 50  # grid points along spatial direction (along x-axis)
N = 60  # grid points along time direction

# Define the coordinates
x0 = -40
xL = 40

t0 = 0
tf = 10

dx = (xL - x0) / (M - 1)
dt = (tf - t0) / (N - 1)

xspan = np.linspace(x0, xL, M)
tspan = np.linspace(t0, tf, N)

# parameter
#alpha = 1.7     # D - Thermal diffusivity
w = 0.2
k = w**2
lamb = 10.
alpha = k**(1/4)

r = dt*alpha/dx**2


# Creating matrix
main_diag = (1 + 2*r)*np.ones((1, M-2))
off_diag = -r*np.ones((1, M-3))

a = main_diag.shape[1]

diagonals = [main_diag, off_diag, off_diag]

A = sparse.diags(diagonals, [0, -1, 1], shape=(a, a)).toarray()     # creates the tridiagonal matrix that's sparse (has many zeros)
print(A)

def valovna_prvi(x):
    global alpha, lamb
    Psi0 = np.sqrt(alpha*np.sqrt(np.pi)) * np.exp( -0.5*(alpha*(x - lamb))**2 )  
    return Psi0

# Initializing matrix and adding Initial conditions (initial function)
T = np.zeros((M, N))
#T[:, 0] = 4*xspan - 4*xspan**2      # initial condition
T[:, 0] = valovna_prvi(xspan)      # initial condition

# Boundary conditions
T[0, :] = 0.0   # left BC
T[-1, :] = 0.0   # right BC


# Solving the Matrix
for k in range(1, N):
    c = np.zeros((M-4, 1)).ravel()
    b1 = np.asarray([r*T[0, k], r*T[-1, k]])
    b1 = np.insert(b1, 1, c)
    b2 = np.array(T[1:M-1, k-1])
    b = b1 + b2     # Right hand side vector
    # AT_{j+1}  T_j + b_j
    T[1:M-1, k] = np.linalg.solve(A, b) # Solving the set of equations
    
    
# Plotting
X, Y = np.meshgrid(tspan, xspan)

fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')
#ax = fig.gca()

surf = ax.plot_surface(X, Y, T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


#ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2])

ax.set_xlabel('Space')
ax.set_ylabel('Time')
ax.set_zlabel('T')
ax.view_init(elev=33, azim=36)

plt.tight_layout()
plt.show()