import numpy as np
import matplotlib.pyplot as plt
import os
import sys


from diffeq import *
from bvp import *


def potential_well(x, a, b, E):
    if a <= x <= b:
        return E
    else:
        return V

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
    #x = np.linspace(-A/2, A/2, N)
    #x = np.linspace(-A/2, A/2, N)
    
    if k % 2 == 0:
        return np.sin(k/A*np.pi*x)  
    else:
        return np.cos(k/A*np.pi*x)

def matrika_neskoncen(n):
    A = np.zeros((n, n))
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:, 1:], 1)
    np.fill_diagonal(A, -2)
    A[0, 1] = 1
    A[-1, -2] = 1

    return A


# Schrodinger parameters
E = 5.0
V = 10.0
#V = float('inf')
a = -5.0    # - solution value at the left boundary: a = y(t[0])
b = 5.0     # - solution value at the right boundary: b = y(t[n-1])
k = 1.0

#velikost in interval a na x osi 
stvr = 5 # number to present on graph
N = 100
a = 10
A = matrika_neskoncen(N)
eigval, eigvec = np.linalg.eigh(A)

x = np.linspace(-a/2, a/2, N)


# Samo vektorji**2 analiticni pa navadno
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.set_title('Analitične $|\\psi_n|^2$')
for i in range(5):
    x = np.linspace(-a/2, a/2, N)
    anal = (analytic(x, i+1, a))**2*2/a
    ax1.plot(x, anal, label=f'n = {i}')
ax1.set_xlabel("x")
ax1.set_ylabel('$|\\psi_n|^2$')
ax1.legend()

ax2.set_title('$|\\psi_n|^2$ z diferenčno metodo')
for i in range(5):
    vektorji = np.abs(eigvec[:, i]) ** 2 * N / a
    #strel = shoot(schrodinger_function, 0, 0, 1, 1, x, 1e-3)
    #print(strel)
    #vektorji = np.abs(strel[i]) ** 2 * N / a
    ax2.plot(x, vektorji, label=f'n = {i}')
ax2.set_xlabel("x")
ax2.set_ylabel('$|\\psi_n|^2$')
ax2.legend()

plt.tight_layout()
plt.show()
plt.clf()
plt.close()


# vektorji premaknjeni plus jama plus energija
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

amplitude = 5
ax1.set_title('Analitične $|\\psi_n|^2$')

#ax1.axhline(y=0, xmin=-a, xmax=-a/2, c='k', ls='--', lw=0.8, alpha=0.8)
#ax1.axhline(y=0, xmin=a/2, xmax=a, c='k', ls='--', lw=0.8, alpha=0.8)
#ax1.axvline(x=-a/2, ymin=-V, ymax=0, c='k', ls='--', lw=0.8, alpha=0.8)
#ax1.axvline(x=a/2, ymin=-V, ymax=0, c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([-a, -a/2], [V, V], c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([a/2, a], [V, V], c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([-a/2, a/2], [0, 0], c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([-a/2, -a/2], [0, V], c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([a/2, a/2], [0, V], c='k', ls='--', lw=0.8, alpha=0.8)
for i in range(stvr):
    x = np.linspace(-a/2, a/2, N)
    anal = (analytic(x, i+1, a))**2*2/a *amplitude + i*V/stvr
    ax1.plot(x, anal, label=f'n = {i}')
ax1.set_xlabel("x")
ax1.set_ylabel('$|\\psi_n|^2$')
ax1.legend()
ax1.set_xlim(-7, 7)


ax2.set_title('$|\\psi_n|^2$ z diferenčno metodo')

ax2.axhline(y=0, xmax=-a/2, c='k', ls='--', lw=0.8, alpha=0.8)
ax2.axhline(y=0, xmin=a/2, c='k', ls='--', lw=0.8, alpha=0.8)
ax2.axvline(x=-a/2, ymin=-V, ymax=0, c='k', ls='--', lw=0.8, alpha=0.8)
ax2.axvline(x=a/2, ymin=-V, ymax=0, c='k', ls='--', lw=0.8, alpha=0.8)
for i in range(stvr):
    vektorji = (eigvec[:, i]) * np.sqrt(N / (a)) *amplitude + i*V/stvr
    ax2.plot(x, vektorji, label=f'n = {i}')
ax2.set_xlabel("x")
ax2.set_ylabel('$|\\psi_n|^2$')
ax2.legend()

plt.tight_layout()
plt.show()



# vektorji
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.set_title('Analitične $|\\psi_n|^2$')
for i in range(5):
    x = np.linspace(-a/2, a/2, N)
    anal = (analytic(x, i+1, a))*2/a
    ax1.plot(x, anal, label=f'n = {i}')
ax1.set_xlabel("x")
ax1.set_ylabel('$|\\psi_n|^2$')
ax1.legend()

ax2.set_title('$|\\psi_n|^2$ z diferenčno metodo')
for i in range(5):
    diferencne = (eigvec[:, i]) * N / (a)
    ax2.plot(x, diferencne, label=f'n = {i}')
ax2.set_xlabel("x")
ax2.set_ylabel('$|\\psi_n|^2$')
ax2.legend()

plt.tight_layout()
#plt.show()