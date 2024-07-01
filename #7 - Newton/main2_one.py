import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as color
import time
import psutil
import scipy.special

from diffeq_2 import *

import threading

def f( x, t ):
    return np.array([x[1], -np.sin(x[0])])

def newt( x ):
    return -np.sin(x)

def exact(t, x0, w0):
    return 2*np.arcsin(np.sin(x0/2) *
                       scipy.special.ellipj(scipy.special.ellipk((np.sin(x0/2))**2) - w0*t, (np.sin(x0/2)**2))[0])

def analytical_matpend(t, x0, w=1.):
    m = numpy.sin(x0*0.5)
    return (2*numpy.arcsin(m * scipy.special.ellipj(scipy.special.ellipk(m**2) - w*t, m**2.)[0]))


def integrate(method, p0, h):
    x0, v0 = p0
    n = int((b-a)/np.abs(h))
    t = np.linspace( a, b, n )

    if method == rk45:
        values, _ = method(f, p0, t)
    elif (method == verlet) or (method == pefrl):
        values = method(newt, x0, v0, t)
    else:
        values = method(f, p0, t)
    return t, values


a, b = ( 0.0, 20.0 )

w0 = 1.0
v0 = 0.0
x0 = 30.0
x0 = x0*np.pi/180 # radians
h = 1e-4
n = int((b-a)/h)
t = np.linspace( a, b, n )
p0 = np.array([x0, 0])

x_analyt = exact(t, x0, w0)
#x_analyt = analytical_matpend(t, x0, w0)

h_sez = [2.0, 1.0, 0.5, 0.1, 0.001, 1e-4]
h_sez = [1.0, 0.1, 1e-3]
#h_sez = np.linspace(0.02, 0.2, 100) # za non log scale plot
#h_sez = np.logspace(np.log10(3), np.log10(1e-5), 100) # za log scale plot


# Function to perform calculations in a thread
def worker(h, time_results, x_results, v_results, index):
    
    ti, result = integrate(euler, p0, h)
    
    # Lock to ensure thread-safe access to the result_array
    with lock:
        #time_results[index] = ti
        #int_results[index] = result

        time_results.append(ti)
        x_results.append(result[:, 0])
        v_results.append(result[:, 1])

    #print(f"h {h}: Result {result}")
    print(f"h {h}")


# Array to store results in order
#time_results = np.zeros_like(h_sez)
#int_results = np.zeros_like(h_sez)
time_results = []
x_results = []
v_results = []
#err_results = [np.abs(int_results[i] - analyt(t, x0)) for i, t in enumerate(time_results)]



# Lock for thread-safe access to the results array
lock = threading.Lock()

# Create threads
threads = []
for i, h in enumerate(h_sez):
    thread = threading.Thread(target=worker, args=(h, time_results, x_results, v_results, i))
    threads.append(thread)

# Start the threads
for thread in threads:
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

# Print the final results array
#print("Final Results:", results)

cmap = plt.get_cmap('viridis_r')
cmap2 = plt.get_cmap('magma_r')
barve = [cmap(value) for value in np.linspace(0, 1, len(h_sez))]
barve2 = [cmap2(value) for value in np.linspace(0, 1, len(h_sez))]

#print("len: ", len(time_results), len(int_results), len(err_results))
#print(time_results)
#print(int_results)
#time_results = np.array(time_results)
#int_results = np.array(int_results)



#plt.figure(figsize=(12, 6))
fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), ncols=2)

ax1.plot(t, x_analyt, ls='--', c='blue', alpha=0.8, lw=0.8, label='Analytical')
for i, h in enumerate(h_sez):#h_sez[::-1]):
    print(i, h)

    #plt.subplot( 1, 2, 1 )
    #print(x_results[i])
    ax1.plot(time_results[i], x_results[i], ls='-', c=barve[-i])#, alpha=(1-h*0.1), lw=(1-h*0.1), label=f'h = {h}')
    #ax1.set_yscale('log')
    ax1.set_xlabel(r'time t [s]')
    ax1.set_ylabel(r'Amplituda')
    ax1.set_title(r'Rezultati za različne velikosti koraka')
    #ax1.legend()

    xh_analyt = exact(t, x0, w0)
    #plt.subplot( 1, 2, 2 )
    ax2.plot(x_results[i], v_results[i], ls='-', c=barve[-i])#, alpha=(1-h*0.1), lw=(1-h*0.1), label=f'h = {h}')

    #ax2.plot(time_results[i], np.abs(int_results[i, 0] - xh_analyt(time_results[i], x0)), ls='-', c=barve[i])#, alpha=(1-h*0.1), lw=(1-h*0.1), label=f'h = {h}')
    #ax2.set_yscale('log')
    ax2.set_xlabel(r'time t [s]')
    ax2.set_ylabel(r'$\Delta\varepsilon$')
    ax2.set_title(r'Absolutna napaka')
    #plt.legend()

ax1.legend()
    
# Add a colorbar
segments = [np.column_stack((x, y)) for x, y in zip(time_results, x_results)]
#print(segments)
col = LineCollection(segments, cmap="viridis_r", norm=color.Normalize())
col.set_array(h_sez)
ax1.add_collection(col, autolim=True)

# Create a colorbar with LogNorm on the second axis
cbar = plt.colorbar(col, ax=ax2, label="Velikost koraka h", norm=color.LogNorm())
cbar.set_label(r"Velikost koraka integracije - $h$")

plt.tight_layout()
plt.show()
