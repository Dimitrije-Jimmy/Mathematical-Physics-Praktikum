import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.linalg import solve_banded

def Bspline(xsez, x):
    B = np.zeros((len(x), len(xsez)))

    dx = xsez[1] - xsez[0]

    for k in range(-1, len(xsez) - 2):
        mask_1 = (x <= xsez[k-2])
        mask_2 = (xsez[k-2] <= x) & (x <= xsez[k-1])
        mask_3 = (xsez[k-1] <= x) & (x <= xsez[k])
        mask_4 = (xsez[k] <= x) & (x <= xsez[k+1])
        mask_5 = (xsez[k+1] <= x) & (x <= xsez[k+2])
        mask_6 = (xsez[k+2] <= x)

        B[mask_1, k] = 0
        B[mask_2, k] = ((x[mask_2] - xsez[k-2])/dx)**3
        B[mask_3, k] = ((x[mask_3] - xsez[k-2])/dx)**3 - 4*((x[mask_3] - xsez[k-1])/dx)**3
        B[mask_4, k] = ((xsez[k-2] - x[mask_4])/dx)**3 - 4*((xsez[k-1] - x[mask_4])/dx)**3
        B[mask_5, k] = ((xsez[k-2] - x[mask_5])/dx)**3
        B[mask_6, k] = 0

    return B


def matrikaA(n):
    A = np.zeros((n, n))
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:, 1:], 1)
    np.fill_diagonal(A, 4)
    A[0, 1] = 1
    A[-1, -2] = 1

    return A

def matrikaB(n):
    global D, a, b, dx
    dx = int(b-a)/n
    
    B = np.zeros((n, n))
    np.fill_diagonal(B[1:], 1)
    np.fill_diagonal(B[:, 1:], 1)
    np.fill_diagonal(B, -2)
    B[0, 1] = 1
    B[-1, -2] = 1

    B = 6*D/dx*B
    return B


a, b, n = 0., 1., 100
D = 0.05
dt = 0.01
n_steps = 100

# Initial condition: sin(pi*x)
x_values = np.linspace(a, b, n)
f_initial = np.sin(np.pi * x_values)

# Construct matrices A and B
A = matrikaA(n)
B = matrikaB(n)

# Set up the time-stepping loop
c_n = f_initial.copy()

for step in range(n_steps):
    # Construct the right-hand side vector
    rhs = np.dot(A + dt/2 * B, c_n)
    
    """
    # Solve the tridiagonal system using solve_banded
    #ab = np.vstack([B[-1, :], A, B[1, :]])
    ab = np.vstack([B[-1, :], A[:-1, :], B[1, :-1]])
        
    # Ensure the correct size by removing the last row if needed
    if len(rhs) < len(banded_matrix[0]):
        banded_matrix = banded_matrix[:, :-1]
    """
    # Adjust the size of the banded matrices to match the length of c_n
    lower_diag = np.hstack([0, B[1:, 0]])
    upper_diag = np.hstack([B[:-1, -1], 0])
    ab = np.array([lower_diag, A.diagonal(), upper_diag])
        
    #print(ab)
    #print(ab.shape)
    c_n1 = solve_banded((1, 1), ab, rhs)
    
    # Update for the next time step
    c_n = c_n1

# Calculate the temperature distribution T(x, t)
T_xt = np.dot(Bspline(x_values, x_values), c_n)
print(T_xt)

# Plot the results
plt.plot(x_values, T_xt)
plt.xlabel('x')
plt.ylabel('T(x, t)')
plt.title('1D Heat Equation - Temperature Distribution')
plt.show()


# Plotting 3D __________________________________________________________________
# Create a grid of coordinates (X, T)
T, X = np.meshgrid(np.linspace(0, n_steps*dt, n_steps), x_values)

# Initialize a 2D array to store the temperature values
temperature_values = np.zeros_like(X)

# Set initial condition
temperature_values[:, 0] = f_initial

# Time-stepping loop to compute temperature values
for step in range(1, n_steps):
    rhs = np.dot(A + dt/2 * B, temperature_values[:, step-1])
    temperature_values[:, step] = solve_banded((1, 1), [lower_diag, A.diagonal(), upper_diag], rhs)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, T, temperature_values, cmap='viridis', edgecolor='k')

# Customize the plot
ax.set_xlabel('Space (x)')
ax.set_ylabel('Time (t)')
ax.set_zlabel('Temperature (T)')
ax.set_title('3D Plot of Temperature Evolution')

# Add colorbar
fig.colorbar(surf, ax=ax, label='Temperature')

# Show the plot
plt.show()