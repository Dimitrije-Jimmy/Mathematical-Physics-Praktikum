import numpy as np
import matplotlib.pyplot as plt
import os
import sys

#file_dir = os.path.dirname(__file__)+'\\'
#sys.path.append(file_dir)

from diffeq import *
from bvp import *
#from functions import *


def potential_well(x, a, b, E):
    if a <= x <= b:
        return E
    else:
        return 0

def schrodinger_function(psi, x):
    # Schrödingers equation for well potential
    """
    - function dy/dt = f(y,t).  Since we are solving a second-
                order boundary-value problem that has been transformed
                into a first order system, this function should return a
                1x2 array with the first entry equal to y and the second
                entry equal to y'
    """
    global a, b, V, E
    k = 1  # hbar^2/2m
    return np.array([psi[1], 1/k * (potential_well(x, a, b, V) - E) * psi[0]])


def analytic(x, k, A=-5.0):
    if k % 2 == 0:
        return np.sin(k*0.5*np.pi*x)
    else:
        return np.cos(k*0.5*np.pi*x)


# Schrodinger parameters
E = 5.0
V = -10.0
a = -5.0    # - solution value at the left boundary: a = y(t[0])
b = 5.0     # - solution value at the right boundary: b = y(t[n-1])
k = 1.0


# Testing ______________________________________________________________________
def test(x, t):
    return np.array([x[1], x[0] + 4*np.exp(t)])


def test_exact(t):
    return np.exp(t) * (1 + 2*t)
# Test plots
a = -5.0
b = 5.0
tol = 10**-6
z1 = np.exp(a)
z2 = 2*np.exp(b)
t_space = np.linspace(a, b, 300)
sol = shoot(schrodinger_function, z1, z2, z1, z2, t_space, tol)
sol2 = fd(4*np.exp(t_space), 1, 0, t_space, z1, z2)
exact = test_exact(t_space)
exact = analytic(t_space, 1.0)
#
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(t_space, sol, label="Shoot", c="#B7094C")
ax1.plot(t_space, sol2, label="FD", c="#723C70")
ax1.plot(t_space, exact, label="Točna rešitev", c="#0091AD")
ax1.legend()
ax2.plot(t_space, np.abs(sol - exact), label="Shoot", c="#B7094C")
ax2.plot(t_space, np.abs(sol2 - exact), label="FD", c="#723C70")
ax2.legend()
#
plt.suptitle(r"Poskusno reševanje $x'' = x + \exp{t}$")
plt.yscale("log")
ax1.set_xlabel("x")
ax1.set_ylabel(r"$f(x)$")
ax2.set_xlabel("x")
ax2.set_ylabel(r"$|f(x) - exact|$")
ax2.yaxis.tick_right()
plt.show()
# Testing ______________________________________________________________________


