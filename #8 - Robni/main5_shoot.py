import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.optimize import newton

#from diffeq import *
from bvp import *


# Functions ____________________________________________________________________

def rku4( f, x0, t, V, E ):
    """Fourth-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = rku4(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """

    n = len( t )
    x = numpy.array( [ x0 ] * n )
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        k1 = h * f( x[i], t[i], V, E )
        k2 = h * f( x[i] + 0.5 * k1, t[i] + 0.5 * h, V, E )
        k3 = h * f( x[i] + 0.5 * k2, t[i] + 0.5 * h, V, E )
        k4 = h * f( x[i] + k3, t[i+1], V, E )
        x[i+1] = x[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0

    return x

def schrodinger(y, r, V, E):
    dydt = [y[1], (V - E) * y[0]]
    return np.asarray(dydt)


def shoot_psi(f, psi0, x, V, E_arr):
    """"Shooting method: find zeroes of Schrödinger equation f with potential V for energies in array E_arr"""
    psi_right = []
    for energy in E_arr:
        psi = rku4(f, psi0, x, V, energy)
        dim = np.shape(psi)[1]
        psi_right.append(psi[0][dim - 1])

    return np.array(psi_right)


def one_shot(E, f, psi0, x, V):  # Reordered inputs for scipy newton
    """Same as shoot_psi but only for one value of energy"""
    psi = rku4(f, psi0, x, V, E)
    return psi[0][np.shape(psi)[1] - 1]


def find_zeroes(rightbound_vals):
    """Amazing method found online. Find zero crossing due to sign change in input array."""
    return np.where(np.diff(np.signbit(rightbound_vals)))[0]


def optimize_energy(f, psi0, x, V, E_arr):
    shoot_try = shoot_psi(f, psi0, x, V, E_arr)  # Should now be a 2D array with psi and psi' right bound values
    crossings = find_zeroes(shoot_try)
    energy_list = []
    for i in crossings:
        # Use Newton-Raphson method to find zero of function
        energy_list.append(newton(one_shot, E_arr[i], args=(f, psi0, x, V)))

    return np.array(energy_list)


def potwell(psi_init, upper, h_):
    """Solves infinite potential well numerically and analytically. Also returns eigenenergies."""
    x_arr_ipw = np.arange(0, 1+h_, h_)
    V_ipw = np.zeros(len(x_arr_ipw))
    E_arr = np.arange(1, upper, 5)  # Set initial guesses for eigen energies
    eigE = optimize_energy(schrodinger, psi_init, x_arr_ipw, V_ipw, E_arr)
    ipw_output = []
    for energy in eigE:
        # Get numerical solution for each eigen energy
        out = rku4(schrodinger, psi_init, x_arr_ipw, V_ipw, energy)
        ipw_output.append(out[0, :]/np.max(out[0, :]))

    # Get analytical solutions
    k = np.arange(1, len(eigE) + 1)
    ipw_solve_analytical = []
    for kk in k:
        ipw_solve_analytical.append(np.sin(kk*np.pi*x_arr_ipw))

    return x_arr_ipw, np.array(ipw_output), np.array(ipw_solve_analytical), eigE

# Finite _______________________________________________________________
def one_shot_fin(E, f, psi0, x, V):  # Reordered inputs for scipy newton
    """Same as shoot_psi but only for one value of energy"""
    psi = rku4(f, psi0, x, V, E)
    dim = np.shape(psi)[1]
    return psi[0, dim - 1]


def shoot_psi_fin(f, psi0, x, V, E_arr):
    """"Shooting method: find zeroes of Schrödinger equation f with potential V for energies in array E_arr"""
    psi_right = []
    mult_psi = []
    for energy in E_arr:
        psi = rku4(f, psi0, x, V, energy)
        dim = np.shape(psi)[1]
        psi_right.append(psi[0, dim - 1])
        mult_psi.append(psi)

    return np.array(psi_right)


def optimize_energy_fin(f, psi0, x, V, E_arr):
    shoot_try = shoot_psi_fin(f, psi0, x, V, E_arr)
    # TODO: Fix boundry value here for 0
    crossings = find_zeroes(shoot_try)
    energy_list = []
    for i in crossings:
        # Use Newton-Raphson method to find zero of function
        energy_list.append(newton(one_shot_fin, E_arr[i], args=(f, psi0, x, V)))

    return np.array(energy_list)


def finpotwell(psi_init, upper, V, h):
    """Solves finite potential well numerically. Also returns eigenenergies."""
    x_arr_fpw = np.arange(-10, 10 + h, h)
    dim = len(x_arr_fpw)
    pos = int(dim//2.2)
    width = int(2*(dim/2 - pos))
    V_fpw = np.zeros(dim)
    V_fpw[:pos] = V
    V_fpw[(pos + width):] = V
    E_arr = np.arange(1, upper, 5)  # Set initial guesses for eigen energies
    eigE = optimize_energy_fin(schrodinger, psi_init, x_arr_fpw, V_fpw, E_arr)
    fpw_output = []
    for energy in eigE:
        # Get numerical solution for each eigen energy
        out = rku4(schrodinger, psi_init, x_arr_fpw, V_fpw, energy)
        fpw_output.append( out[0, :]/np.max(out[0, :][pos:(pos + width)]) )

    return x_arr_fpw, np.array(fpw_output), eigE, V_fpw



# Plotting _____________________________________________________________________

# Set initial conditions and parameters
psi_0 = 0
phi_0 = 1  # psi' = phi
psi_init = np.array([psi_0, phi_0])
h = 1/2000  # Step size
upper = 500
depth = 100
t = np.arange(-10, 10+h, h)

# Finding zeros ________________________________________________________


# Neskončna jama _______________________________________________________



# Končna jama __________________________________________________________




