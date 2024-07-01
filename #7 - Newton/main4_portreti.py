import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.integrate import odeint

# Define the system of first-order differential equations
def lotka_volterra(y, t, alpha, beta, delta, gamma):
    x, y = y
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

def f( x ):
    return np.array([x[1], -np.sin(x[0])])

def f2( x, beta=0.5, w0=2./3. ):
    return np.array([x[1], -beta*x[1] - np.sin(x[0]) + w0*x[1]])

def f3( x , t=0., w0=1., v0=10., lamb=1.):
    return np.array([x[1], v0*np.cos(w0*t) + lamb*x[1]*(1 - x[0]**2) - x[0]])





# Set up the grid
x = np.linspace(-4, 4, 20)
y = np.linspace(-2, 3, 20)
X, Y = np.meshgrid(x, y)
"""
y1 = np.linspace(-8.0, 10.0, 40)
y2 = np.linspace(-10.0, 10.0, 20)
X, Y = np.meshgrid(y1, y2)
"""
# Parameters for the Lotka-Volterra model
#alpha, beta, delta, gamma = 0.1, 0.02, 0.01, 0.1

# Compute the derivatives at each point in the grid
#DX, DY = lotka_volterra([X, Y], 0, alpha, beta, delta, gamma)
DX, DY = f3([X, Y])
M = np.hypot(DX, DY) 

#magnitude = np.sqrt(DX**2 + DY**2)
#DX /= magnitude 
#DY /= magnitude

# Plot the vector field with a manual color gradient
fig, ax = plt.subplots(figsize=(8, 5))


q = ax.quiver(X, Y, DX, DY, M, cmap='winter', scale=20, pivot='mid')

# Add a colorbar for reference
cbar = plt.colorbar(q)
cbar.set_label('Magnitude')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Vzbujeno matematično nihalo')
plt.show()


import sys
#sys.exit()
# VanDerPool ___________________________________________________________________
y1 = np.linspace(-8.0, 10.0, 40)
y2 = np.linspace(-16.0, 16.0, 20)

Y1, Y2 = np.meshgrid(y1, y2)

t = 0

u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)

NI, NJ = Y1.shape

for i in range(NI):
    for j in range(NJ):
        x = Y1[i, j]
        y = Y2[i, j]
        yprime = f3([x, y])
        u[i, j] = yprime[0]
        v[i, j] = yprime[1]
        
M = np.hypot(u, v) 
Q = plt.quiver(Y1, Y2, u, v, M, cmap='viridis_r')

plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.xlim([-8, 8])
plt.ylim([-16, 16])

plt.show()
