import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as color

from diffeq import *

def f( x, t, k=0.1, T_out = -5 ):
#        return x * numpy.sin( t )
    #return t * numpy.sin( t )
    return -k*(x - T_out)


a, b = ( 0.0, 21.0 )



T0 = np.arange(-20.0, 25.0, 3)
k_sez = np.linspace(0.01, 5, len(T0))
T_out = -5.0
k = 0.1

h = 1e-3
n = int((b-a)/np.abs(h))
t = np.linspace( a, b, n )

# 3D spreminjanje T0 ___________________________________________________________
def integrate_and_plot(method, f, T0, t):
    #T0 = np.array(T0)
    t_grid, T0_grid = np.meshgrid(t, T0)
    #print(t)
    #print(T0)

    # Initialize an array to store the results
    results = np.zeros_like(T0_grid)

    # Perform integration for each set of initial conditions
    for i in range(len(T0)):
        print(f'step: {i}')
        result = method(f, T0[i], t)

        results[i] = result


    # Create a 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    
    # Plot the surface
    surf = ax.plot_surface(t_grid, T0_grid, results, cmap='viridis', edgecolor='k')

    # Customize the plot
    ax.set_xlabel(r'Čas [$s$]')
    ax.set_ylabel(r'Začetna temperatura $T_0$ [$^{\circ}$C]')
    ax.set_zlabel(r'Temperatura $T(t)$ [$^{\circ}$C]')
    ax.set_title(r'Graf T(t) za različne začetne pogoje')

    # Add colorbar for reference
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # Show the plot
    plt.tight_layout()
    plt.show()


# 3D spreminjanje k ____________________________________________________________
def integrate_and_plot2(method, f, k_sez, t):
    #T0 = np.array(T0)
    t_grid, k_grid = np.meshgrid(t, k_sez)
    #print(t)
    #print(T0)

    # Initialize an array to store the results
    results = np.zeros_like(k_grid)
    

    # Perform integration for each set of initial conditions
    for i in range(len(k_sez)):
        print(f'step: {i}')

        def f2( x, t, k=0.1, T_out = -5 ):
            return -k_sez[i]*(x - T_out)

        result = method(f2, 21.0, t)

        results[i] = result


    # Create a 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    
    # Plot the surface
    surf = ax.plot_surface(t_grid, k_grid, results, cmap='viridis')#, norm=color.LogNorm())

    # Customize the plot
    ax.set_xlabel(r'Čas [$s$]')
    ax.set_ylabel(r'Vrednost parametra $k$')
    ax.set_zlabel(r'Temperatura $T(t)$ [$^{\circ}$C]')
    ax.set_title(r'Graf T(t) za različne vrednosti $k$')

    # Add colorbar for reference
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage:
integrate_and_plot(euler, f, T0, t)

integrate_and_plot2(euler, f, k_sez, t)