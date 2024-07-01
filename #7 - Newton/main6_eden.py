import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as color

import threading
import time
import psutil
import scipy.special

from scipy.integrate import odeint
from diffeq_2 import *

def funkcija( x, t ):
    return np.array([x[1], -np.sin(x[0])])

def newt( x ):
    return -np.sin(x)


def exact(t, x0, w0):
    return 2*np.arcsin(np.sin(x0/2) *
                       scipy.special.ellipj(scipy.special.ellipk((np.sin(x0/2))**2) - w0*t, (np.sin(x0/2)**2))[0])

def energy(x_sez, v_sez, w0):
    return 1 - np.cos(x_sez) + (v_sez**2)/(2*w0**2)

def integrate(method, f, p0, h):
    n = int((b-a)/np.abs(h))
    t = np.linspace( a, b, n )
    #t = np.arange( a, b, h )
    
    x0, v0 = p0

    if method == rk45:
        values, _ = method(funkcija, p0, t)
        x, v = values[:, 0], values[:, 1]
    elif method == rkf:
        t, values = method(funkcija, a, b, p0, 10, 0.3, 1e-6)
        x, v = values[:, 0], values[:, 1]
    elif method == odeint:
        values, _ = odeint(funkcija, p0, t, full_output=1)
        #values = odeint(funkcija, p0, t, full_output=1)
        x, v = values[:, 0], values[:, 1]
    elif method == verlet:
        values = verlet(newt, x0, v0, t)
        x, v = values
    elif method == pefrl:
        values = pefrl(newt, x0, v0, t)
        x, v = values
    else:
        values = method(funkcija, p0, t)
        x, v = values[:, 0], values[:, 1]
    return t, x, v

a, b = ( 0.0, 35.0 )

w0 = 1.0
v0 = 0.0
x0 = 1.0
h = 1e-2
n = int((b-a)/h)
t = np.linspace( a, b, n )
x_analyt = exact(t, x0, w0)
p0 =  np.array([x0, v0])

h_sez = [2.0, 1.0, 0.5, 0.1, 0.001]#, 1e-4]
#h_sez = np.linspace(0.001, 0.3, 500) # za non log scale plot
#h_sez = np.logspace(np.log10(3), np.log10(1e-5), 100) # za log scale plot

h_sez = [1.0, 0.1, 0.01]
#h_sez = np.around(np.linspace(0.001, 2.0, 50), 4)
h_sez = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]

# Number of additional points between each pair
num_additional_points = 20

# Fill in additional points
new_h_sez = np.hstack([np.linspace(start, stop, num_additional_points, endpoint=False) for start, stop in zip(h_sez[:-1], h_sez[1:])])
# Combine the original array and the additional points
h_sez = np.concatenate([h_sez, new_h_sez, [h_sez[-1]]])

#h_sez = [0.1, 0.05, 0.01, 0.005]


def worker(h, time_results, x_results, v_results, exec_results, 
           xerr_results, verr_results, methods, index):
    global x0, v0, w0

    start_time = time.process_time()
    ti, x, v = integrate(euler, funkcija, p0, h)
    execution_time = time.process_time() - start_time  
    
    xerr = (np.abs(x - exact(ti, x0, w0)))#/len(x)
    verr = (np.abs(v - exact(ti, x0, w0)))#/len(v)    
    
    # Lock to ensure thread-safe access to the result_array
    with lock:
        #print(exec_vmesni)
        
        time_results[index] = ti
        x_results[index] = x
        v_results[index] = v
        exec_results[index] = execution_time
        xerr_results[index] = xerr
        verr_results[index] = verr

    #print(f"h {h}: Result {result}")
    print(f"h {h}")


# Array to store results in order
time_results = np.zeros_like(h_sez, dtype=object)
x_results = np.zeros_like(h_sez, dtype=object)
v_results = np.zeros_like(h_sez, dtype=object)
exec_results = np.zeros_like(h_sez, dtype=object)
xerr_results = np.zeros_like(h_sez, dtype=object)
verr_results = np.zeros_like(h_sez, dtype=object)


#methods = [euler, rk45, odeint, verlet, pefrl]
#methods_names = ['Euler', 'Runge-Kutta 45', 'Odeint', 'Verlet', 'PEFRL']

#methods = [euler, rk45, verlet, pefrl]
#methods_names = ['Euler', 'Runge-Kutta 45', 'Verlet', 'PEFRL']

methods = [euler]
methods_names = ['Euler']

# Lock for thread-safe access to the results array
lock = threading.Lock()

# Create threads
threads = []
for index, h in enumerate(h_sez):
    thread = threading.Thread(target=worker, args=(h, time_results, x_results, v_results, exec_results, xerr_results, verr_results, methods, index))
    threads.append(thread)

# Start the threads
for thread in threads:
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()
"""
time_results = np.stack(time_results, dtype=numpy.dtype)
x_results = np.stack(x_results, dtype=numpy.dtype)
v_results = np.stack(v_results, dtype=numpy.dtype)
exec_results = np.stack(exec_results, dtype=numpy.dtype)
xerr_results = np.stack(xerr_results, dtype=numpy.dtype)
verr_results = np.stack(verr_results, dtype=numpy.dtype)
"""

#print(time_results)
#print(xerr_results)

# Plotting _____________________________________________________________________
cmap = plt.get_cmap('viridis')
#cmap = plt.get_cmap('winter')
cmap2 = plt.get_cmap('magma_r')
barve = [cmap(value) for value in np.linspace(0, 1, len(h_sez))]
barve2 = [cmap2(value) for value in np.linspace(0, 1, len(h_sez))]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#a93226']
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(figsize=(15, 8), nrows=1, ncols=2)#, facecolor='darkgrey')


#plt.subplot( 1, 2, 1 )
ax1.plot(t, x_analyt, ls='--', lw=1.0, c='white', label='exact')
for i in range(len(h_sez)):
    print(i)
    #ax1.plot(h_sez, exec_results[:, i], ls='-', c=colors[i], label=f'{methods_names[i]}')
    #ax1.plot(h_sez, exec_results[:, i], ls='-')#, lw=1.0, alpha=0.9)
    #ax1.scatter(h_sez, exec_results[:, i], s=1.5, alpha=0.9, c=colors[i], label=f'{methods_names[i]}')
    #ax1.scatter(time_results[i], x_results[i], c=barve[i])#, label=f'{methods_names[i]}')
    ax1.plot(time_results[i], x_results[i], ls='-', c=barve[i], lw=1.0, alpha=0.9)
#ax1.set_xlim(0.0, 35.0)
#ax1.set_ylim(-4.0, 4.0)
#ax1.set_yscale('log')
#ax1.set_xscale('log')
ax1.set_xlabel(r'Čas $t$ [$s$]')
ax1.set_ylabel(r'Amplituda')
ax1.set_title(r'Različni koraki - $h$')
ax1.legend()
#ax1.set_facecolor((0, 0, 0, 0.7))

#plt.subplot( 1, 2, 2 )
#ax2.axhline(y=1e-3, ls='--', lw=0.8, c='k')
for i in range(len(h_sez)):
    print(i)
    #ax2.plot(h_sez, xerr_results[:, i], ls='-', c=colors[i], label=f'{methods_names[i]}')
    #ax2.plot(h_sez, xerr_results[:, i], ls='-')#, lw=1.0, alpha=0.9)
    #ax2.scatter(h_sez, xerr_results[:, i], s=1.5, alpha=0.9, c=colors[i], label=f'{methods_names[i]}')
    q = ax2.plot(time_results[i], xerr_results[i], c=barve[i], lw=1.0, alpha=0.9)
#ax2.set_xlim(0.0, 35.0)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'Čas $t$ [$s$]')
ax2.set_ylabel(r'Globalna absolutna napaka - $\Delta\varepsilon$')
ax2.set_title(r'Absolutna napaka')
#ax2.legend()
#ax2.set_facecolor((0, 0, 0, 0.7))
# Add a colorbar
segments = [np.column_stack((x, y)) for x, y in zip(time_results, x_results)]
#print(segments)
col = LineCollection(segments, cmap="viridis_r")#, norm=color.Normalize())
col.set_array(h_sez)
#ax1.add_collection(col, autolim=True)

# Create a colorbar with LogNorm on the second axis
cbar = plt.colorbar(col, ax=ax2, label="Velikost koraka h")#, norm=color.LogNorm())
cbar.set_label(r"Velikost koraka - $h$")


plt.tight_layout()
plt.show()

# Absolutna napaka v odvisnosti od execution časa + step size
fig, (ax1) = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
plt.style.use('dark_background')

cmap1 = plt.get_cmap('viridis_r')
cmap2 = plt.get_cmap('plasma')
cmap3 = plt.get_cmap('RdYlBu')
cmap4 = plt.get_cmap('winter')
cmap5 = plt.get_cmap('twilight')
colormaps = [cmap1, cmap2, cmap3, cmap4, cmap5]

barve = [cmap1(value) for value in np.linspace(0, 1, len(h_sez))]
for i in range(len(h_sez)):
    print(i)
    
    #ax1.plot(exec_results[:, i], xerr_results[:, i], ls='-', lw=1.0, alpha=0.5, label=f'{methods_names[i]}')
    q = ax1.scatter(exec_results[i], np.sum(xerr_results[i])/len(xerr_results[i]), c=barve[i])#, label=f'{methods_names[i]}')
    #ax1.plot(xerr_results[:, i], exec_results[:, i], ls='-', lw=1.0, alpha=0.9, label=f'{methods_names[i]}')
    #q = ax1.scatter(xerr_results[:, i], exec_results[:, i], c=h_sez, cmap=colormaps[i])
    
# Add colorbar for reference
cbar = plt.colorbar(q, cmap='viridis_r')#, shrink=0.6, pad=0.05)
cbar.set_label(r"Velikost koraka - $h$")


ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'Čas izvajanja [$s$]')
ax1.set_ylabel(r'Absolutna napaka')
ax1.set_title(r'Željena natančnost v odvisnosti od časa izvajanja')
#ax1.legend()

plt.tight_layout()

plt.show()

import sys
sys.exit()
# za različne korake $h$ _______________________________________________________

fig, (ax1) = plt.subplots(figsize=(15, 8), nrows=1, ncols=1)

cmap = plt.get_cmap('plasma_r')

for i, h in enumerate(h_sez):
    print(i, h)
    
    ax1.plot(xerr_results[:, i], exec_results[:, i], ls='-', lw=1.0, alpha=0.9)

    

#ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.set_xlabel(r'Čas izvajanja [$s$]')
ax1.set_ylabel(r'Amplituda')
ax1.set_title(r'Rešitve za različne vrednosti $h$')
ax1.legend()

plt.tight_layout()  
    
    
plt.show()

