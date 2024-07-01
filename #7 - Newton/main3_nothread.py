import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as color
import time
import psutil
import scipy.special

from scipy.integrate import odeint
from diffeq_2 import *

def f( x, t ):
    return np.array([x[1], -np.sin(x[0])])

"""
def f2( x, t, beta=0.5, w0=2./3. ):
    return np.array([x[1], -beta*x[1] - np.sin(x[0]) + w0*np.cos(w0*t)])

def f22( x, beta=0.5, w0=2./3.):
    return -beta*x[1] - np.sin(x[0]) + w0*np.cos(w0*t)

def f3( x , t=0., w0=1., v0=10., lamb=1.):
    return np.array([x[1], v0*np.cos(w0*t) + lamb*x[1]*(1 - x[0]**2) - x[0]])
"""

def newt( x ):
    return -np.sin(x)


def exact(t, x0, w0):
    return 2*np.arcsin(np.sin(x0/2) *
                       scipy.special.ellipj(scipy.special.ellipk((np.sin(x0/2))**2) - w0*t, (np.sin(x0/2)**2))[0])

def energy(x_sez, v_sez, w0):
    return 1 - np.cos(x_sez) + (v_sez**2)/(2*w0**2)
    

a, b = ( 0.0, 100 )

w0 = 1.0
v0 = 0.0
#x0 = 30.0
#x0 = x0*np.pi/180 # radians
x0 = 1.0
h = 1e-2
n = int((b-a)/h)
t = np.linspace( a, b, n )
x_analyt = exact(t, x0, w0)
p0 =  np.array([x0, v0])

sol1 = euler(f, p0, t)
sol2, _ = rk45(f, p0, t)
sol3 = verlet(newt, x0, v0, t)
sol4 = pefrl(newt, x0, v0, t)
sol5 = odeint(f, p0, t)


fig, (ax1, ax2, ax3) = plt.subplots(figsize=(8, 15), nrows=3, ncols=1)

ax1.plot(t, x_analyt, ls='--', c='blue', alpha=0.8, lw=0.8, label='Analytical')
ax1.plot(t, sol1[:, 0], ls='-', label='euler')
ax1.plot(t, sol2[:, 0], ls='-', label='rk45')
ax1.plot(t, sol5[:, 0], ls='-', label='odeint')
ax1.plot(t, sol3[0], ls='-', label='verlet')
ax1.plot(t, sol4[0], ls='-', label='pefrl')
ax1.set_xlabel(r'time t [s]')
ax1.set_ylabel(r'Amplituda')
ax1.set_title(r'Rezultati za različne metode pri $h = 10^{-2}$.')
ax1.legend()


#ax2.plot(t, np.abs(sol1[:, 0] - x_analyt)/x_analyt, ls='-')
ax2.plot(t, np.abs(sol1[:, 0] - x_analyt), ls='-', label='euler')
ax2.plot(t, np.abs(sol2[:, 0] - x_analyt), ls='-', label='rk45')
ax2.plot(t, np.abs(sol5[:, 0] - x_analyt), ls='-', label='odeint')
ax2.plot(t, np.abs(sol3[0] - x_analyt), ls='-', label='verlet')
ax2.plot(t, np.abs(sol4[0] - x_analyt), ls='-', label='pefrl')
#ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'time t [s]')
ax2.set_ylabel(r'$\Delta\varepsilon$')
ax2.set_title(r'Absolutna napaka')
ax2.legend()


ax3.plot(sol1[:, 0], sol1[:, 1], ls='-', label='euler')
ax3.plot(sol2[:, 0], sol2[:, 1], ls='-', label='rk45')
ax3.plot(sol5[:, 0], sol5[:, 1], ls='-', label='odeint')
ax3.plot(sol3[0], sol3[1], ls='-', label='verlet')
ax3.plot(sol4[0], sol4[1], ls='-', label='pefrl')

#ax3.set_xlim(-1.0, 1.0)
#ax3.set_ylim(-1.0, 1.0)
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$v$')
ax3.set_title(r'Fazni diagram')
ax3.legend()

plt.tight_layout()
plt.savefig("metode3")

# Energy plot __________________________________________________________________


fig, (ax1, ax2, ax3) = plt.subplots(figsize=(10, 8), nrows=3, ncols=1)

ax1.plot(t, energy(sol1[:, 0], sol1[:, 1], w0), ls='-', label='euler')
ax1.plot(t, energy(sol2[:, 0], sol2[:, 1], w0), ls='-', label='rk45')
ax1.plot(t, energy(sol3[0], sol3[1], w0), ls='-', label='verlet')
ax1.plot(t, energy(sol4[0], sol4[1], w0), ls='-', label='pefrl')
ax1.plot(t, energy(sol5[:, 0], sol5[:, 1], w0), ls='-', label='odeint')
ax1.set_xlabel(r't [s]')
ax1.set_ylabel(r'Energija')
ax1.set_title(r'Energija za različne metode')
ax1.legend()

ax2.plot(t, energy(sol1[:, 0], sol1[:, 1], w0), ls='-', label='euler')
ax2.plot(t, energy(sol2[:, 0], sol2[:, 1], w0), ls='-', label='rk45')
ax2.plot(t, energy(sol3[0], sol3[1], w0), ls='-', label='verlet')
ax2.plot(t, energy(sol4[0], sol4[1], w0), ls='-', label='pefrl')
ax2.plot(t, energy(sol5[:, 0], sol5[:, 1], w0), ls='-', label='odeint')
E0 = energy(sol1[:, 0], sol1[:, 1], w0)[0]
eps = 5e-6
ax2.set_ylim(E0-1.5*eps, E0+0.5*eps)
ax2.set_xlabel(r't [s]')
ax2.set_ylabel(r'Energija')
ax2.set_title(r'Energija za različne metode približana')
ax2.legend()


ax3.plot(t, np.abs(energy(sol1[:, 0], sol1[:, 1], w0) - energy(sol1[:, 0], sol1[:, 1], w0)[0]), ls='-', label='euler')
ax3.plot(t, np.abs(energy(sol2[:, 0], sol2[:, 1], w0) - energy(sol2[:, 0], sol2[:, 1], w0)[0]), ls='-', label='rk45')
ax3.plot(t, np.abs(energy(sol5[:, 0], sol5[:, 1], w0) - energy(sol5[:, 0], sol5[:, 1], w0)[0]), ls='-', label='odeint')
ax3.plot(t, np.abs(energy(sol3[0], sol3[1], w0) - energy(sol3[0], sol3[1], w0)[0]), ls='-', label='verlet')
ax3.plot(t, np.abs(energy(sol4[0], sol4[1], w0) - energy(sol4[0], sol4[1], w0)[0]), ls='-', label='pefrl')
ax3.set_yscale('log')
ax3.set_xlabel(r't [s]')
ax3.set_ylabel(r'$\|E - E_0\|$')
ax3.set_title(r'Absolutna napaka energije')
ax3.legend()

plt.tight_layout()


# Različen x0 __________________________________________________________________

a, b = ( 0.0, 20.0)
N = 7

h = 1e-3
nt = int((b-a)/h)
t = np.linspace(a, b, nt)
theta = np.linspace(0.1, 0.7, N)
v0 = 0.0
w0 = 1.0

fig, (ax1, ax2) = plt.subplots(figsize=(10, 8), nrows=2, ncols=1)


cmap = plt.get_cmap('viridis_r')
cmap = plt.get_cmap('winter_r')
barve = [cmap(value) for value in np.linspace(0, 1, len(theta))]

sol_sez = np.zeros_like(theta, dtype=object)
for i, kot in enumerate(theta):
    sol = pefrl(newt, kot*np.pi, v0, t)
    sol_sez[i] = sol
    
    ax1.plot(t, sol[0]/sol[0, 0], ls='-', c=barve[i], alpha=0.9, lw=0.9, label=f'$\\theta_0$ = {np.around(kot, 2)}$\\pi$')

    ax2.plot(t, energy(sol[0], sol[1], w0), ls='-', c=barve[i], alpha=0.9, lw=0.9, label=f'$\\theta_0$ = {np.around(kot, 2)}$\\pi$')


ax1.set_xlabel(r't [s]')
ax1.set_ylabel(r'Normirana amplituda')
ax1.set_title(r'Različni začetni $\theta_0$')
ax1.legend()
    
ax2.set_xlabel(r't [s]')
ax2.set_ylabel(r'Energija')
ax2.set_title(r'Energija za različne začetne $\theta_0$')
ax2.legend()

plt.tight_layout()
plt.show()