import numpy as np
import matplotlib.pyplot as plt
import time
import os

from scipy import integrate


# Analytical solution __________________________________________________________
"""
T(x, t) = \sum_{n=1}^{\infty} B_n \sin(n\pi x) e^{-n^2\pi^2 t}

B_m = 2\int_0^1 \sin(m \pi x) f(x) dx

f(x) = T(x, 0)
"""

a = 0.
b = 1.
n = 2       # tta rab bit manjša
m = 100
#m = 50
L = 1.0 # dolzina
D = 0.05 # difuzijska konstanta
sigma = L/4. # std
x0 = np.linspace(a, b, m)
t0 = np.linspace(0., 1., m*n)

zp = np.exp(-((x0 - L/2.)**2.)/(sigma**2.))

#plt.plot(x0, zp)
#plt.show()

N = n
M = m
P = 100
def Bkoeficient(zacetna_fun="Gauss"):
    global a, b, N, L, sigma

    if zacetna_fun == "sin":
        integrand = lambda x, i, L: 2*np.sin(i *np.pi * x / L) * np.sin(np.pi * x / L)
        B = np.zeros(P)
        for i in range(P):
            B[i] = 2./L*integrate.quad(integrand, 0., L, args=(i, L))[0]
            
    else:
        #integrand = lambda x, i, L, sigma: 2*np.sin(i *np.pi * x / L) *1/(np.sqrt(2*np.pi)*sigma) *np.exp(-0.5*(x - L/2.)**2. / sigma**2.)
        integrand = lambda x, i, L, sigma: 2*np.sin(i *np.pi * x / L) *np.exp(-(x - L/2.)**2. / sigma**2.)
        B = np.zeros(P)
        for i in range(P):
            B[i] = 2./L*integrate.quad(integrand, 0., L, args=(i, L, sigma))[0]
    
    #print(B)
    return B


def analytic(x0, t0, zacetna_fun):
    global M, N
    B = Bkoeficient(zacetna_fun)
    
    Uxt = np.zeros((M*N, M))
    """for i, tval in enumerate(t0):
        print(i)
        for j, xval in enumerate(x0):
            for k in range(len(B)):
                Uxt[i, j] = Uxt[i, j] + B[k] *np.sin(k *np.pi *xval / L) * np.exp((-1.)*((k * np.pi/L)**2.)*D*tval)
    """
    for i, tval in enumerate(t0):
        print(i)
        Uxt[i, :] = np.sum(
            B[k] * np.sin(k * np.pi * x0 / L) *
            np.exp((-1.) * ((k * np.pi / L)**2.) * D * tval)
            for k in range(len(B))
        )

    return Uxt
    return Uxt
    
zacetna_fun = "Gauss"
#zacetna_fun = "sin"
y = analytic(x0, t0, zacetna_fun)

#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

#print(y)
#print(y[0])

#print(y.shape )
#print(len(y))
#print(len(y[0]))
#print(len(x0))

cmap = plt.get_cmap('hot')
cmap2 = plt.get_cmap('magma_r')

barve1 = [cmap(value) for value in np.linspace(0, 1, len(x0))]
barve2 = [cmap(value) for value in np.linspace(0, 1, len(t0))]

colors = plt.cm.jet(np.linspace(1, 0, M*N))
for yind, yval in enumerate(y):
    print(yind)
    ax1.plot(x0, yval, c=colors[yind])

ax1.set_xlabel('$x$')
ax1.set_ylabel('$T$')
#ax1.plot(x0, y[0])
#ax1.plot(x0, zp)
   
ax2.imshow(np.real(y), cmap='jet', interpolation='nearest')
#ax2.imshow(y, cmap='jet', interpolation='nearest')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$t$')
"""
ax3.imshow(y, cmap='jet', interpolation='nearest')
ax3.set_xlabel('$x$')
ax3.set_ylabel('$t$')
"""

plt.tight_layout()
plt.show()
plt.clf()
plt.close()

import sys
#sys.exit()
# 3D ___________________________________________________________________________

x_grid, t_grid = np.meshgrid(x0, t0)

# Create a 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')


# Plot the surface
surf = ax.plot_surface(x_grid, t_grid, y, cmap='jet', edgecolor='k')

# Customize the plot
ax.set_xlabel(r'x [$^{\circ}$C]')
ax.set_ylabel(r't [$s$]')
ax.set_zlabel(r'Temperatura $T(t)$ [$^{\circ}$C]')
ax.set_title(r'Graf T(x, t) po prostoru in v času')

# Add colorbar for reference
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.tight_layout()
plt.show()

