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

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

a, b, n, m = 0., 1., 2, 100
L, D = 1.0, 0.05
sigma = L/4.
x0 = np.linspace(a, b, m)
t0 = np.linspace(0., 1., m*n)

zp = np.exp(-((x0 - L/2.)**2.)/(sigma**2.))

N, M, P = n, m, 100

def Bkoeficient(zacetna_fun="Gauss"):
    global a, b, N, L, sigma

    integrand = (
        lambda x, i, L, sigma: 2 * np.sin(i * np.pi * x / L) *
        np.exp(-(x - L/2.)**2. / sigma**2.)
    ) if zacetna_fun != "sin" else (
        lambda x, i, L: 2 * np.sin(i * np.pi * x / L) *
        np.sin(np.pi * x / L)
    )

    B = (
        np.array([
        2./L * integrate.quad(integrand, 0., L, args=(i, L, sigma))[0]
        for i in range(P)
        ])
    ) if zacetna_fun != "sin" else (
        np.array([
        2./L * integrate.quad(integrand, 0., L, args=(i, L))[0]
        for i in range(P)
        ])
    )

    return B

def analytic(x0, t0, zacetna_fun='Gauss'):
    global M, N
    B = Bkoeficient(zacetna_fun)
    Uxt = np.zeros((M*N, M))

    for i, tval in enumerate(t0):
        print(i)
        Uxt[i, :] = np.sum(
            B[k] * np.sin(k * np.pi * x0 / L) *
            np.exp((-1.) * ((k * np.pi / L)**2.) * D * tval)
            for k in range(len(B))
        )

    return Uxt

def analytic2(x0, t0, zacetna_fun='Gauss'):
    #global M, N
    M = len(x0)
    print("x0", len(x0))
    N = 2
    B = Bkoeficient(zacetna_fun)
    Uxt = np.zeros((M*N, M))

    for i, tval in enumerate(t0):
        print(i)
        Uxt[i, :] = np.sum(
            B[k] * np.sin(k * np.pi * x0 / L) *
            np.exp((-1.) * ((k * np.pi / L)**2.) * D * tval)
            for k in range(len(B))
        )

    return Uxt

# Execution ____________________________________________________________________
if __name__ == "__main__":

    zacetna_fun = "Gauss"
    zacetna_fun = "sin"
    y = analytic(x0, t0, zacetna_fun)

    # Plotting _____________________________________________________________________
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))


    colors = plt.cm.jet(np.linspace(1, 0, M*N))
    for yind, yval in enumerate(y):
        print(yind)
        ax1.plot(x0, yval, c=colors[yind])

    ax1.plot(x0, y[0], 'k--', label='Začetno stanje')
    ax1.legend()
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$T$')
    #ax1.plot(x0, y[0])
    #ax1.plot(x0, zp)
    ax1.set_title(r'Temperatura v odvisnosti od časa')
    
    surf2 = ax2.imshow(np.flipud(np.real(y)), cmap='jet', interpolation='nearest')
    #ax2.imshow(y, cmap='jet', interpolation='nearest')
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$t$')
    
    """
    # Add colorbar for reference
    cbar2 = fig.colorbar(surf1, ax=ax1, aspect=10)
    cbar2.set_label(r'Čas - $t$ [$s$]')
    """
    # Add colorbar for reference
    cbar2 = fig.colorbar(surf2, ax=ax2, aspect=10)
    cbar2.set_label(r'Temperatura - $T$ [$^{\circ}$C]')
    
    
    
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
    ax.set_xlabel(r'x/L [$m$]')
    ax.set_ylabel(r't [$s$]')
    ax.set_zlabel(r'$T(t)$ [$^{\circ}$C]')
    ax.set_title(r'Temperatura - T(x, t)')

    # Add colorbar for reference
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # Show the plot
    plt.tight_layout()
    plt.show()

