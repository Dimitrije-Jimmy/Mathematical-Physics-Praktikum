import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from scipy.integrate import odeint
import time
import psutil

from scipy.integrate import odeint
from diffeq_2 import *

def f( x, t ):
    return np.array([x[1], -np.sin(x[0])])

def newt( x ):
    return -np.sin(x)


def exact(t, x0, w0):
    return 2*np.arcsin(np.sin(x0/2) *
                       scipy.special.ellipj(scipy.special.ellipk((np.sin(x0/2))**2) - w0*t, (np.sin(x0/2)**2))[0])

def energy(x_sez, v_sez, w0):
    return 1 - np.cos(x_sez) + (v_sez**2)/(2*w0**2)


def integrate(method, f, x0, h):
    n = int((b-a)/np.abs(h))
    t = np.linspace( a, b, n )

    if method == rk45:
        values, _ = method(f, x0, t)
    elif method == rkf:
        t, values = method(f, a, b, x0, 10, 0.3, 1e-3)
    elif method == odeint:
        results, _ = odeint(f, x0, t, full_output=1)
        values = results[:, 0]
    else:
        values = method(f, x0, t)
    return t, values


def exact(t, x0, w0):
    return 2*np.arcsin(np.sin(x0/2) *
                       scipy.special.ellipj(scipy.special.ellipk((np.sin(x0/2))**2) - w0*t, (np.sin(x0/2)**2))[0])

a, b = ( 0.0, 100.0 )

x0 = 30
x0 = x0*np.pi/180 # radians
w0 = 1
t = np.linspace( a, b, 10 )
x_analyt = analyt(t, x0, w0)

h_sez = [2.0, 1.0, 0.5, 0.1, 0.001, 1e-4]
h_sez = np.linspace(0.001, 0.3, 500) # za non log scale plot
#h_sez = np.logspace(np.log10(3), np.log10(1e-5), 100) # za log scale plot
