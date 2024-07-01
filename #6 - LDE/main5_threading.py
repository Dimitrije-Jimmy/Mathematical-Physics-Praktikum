import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as color
import time
import psutil

from diffeq import *

import threading

def f( x, t, k=0.1, T_out=-5.0, A=1, d=10 ):
    #return -k*(x - T_out)
    return -k*(x - T_out) + A*np.sin((2*np.pi/24)*(t - d))

def analyt( t, x0, k=0.1, T_out=-5.0 ):
    return  T_out + np.exp(-k*t)*(x0 - T_out)

def integrate(method, f, x0, h):
    n = int((b-a)/np.abs(h))
    t = np.linspace( a, b, n )

    if method == rk45:
        values, _ = method(f, x0, t)
    else:
        values = method(f, x0, t)
    return t, values


a, b = ( 0.0, 31.0 )

k = 0.1
T_out = -5.0
x0 = -15.0
t = np.linspace( a, b, 10 )
x_analyt = analyt(t, x0)

h_sez = [2.0, 1.0, 0.5, 0.1, 0.001, 1e-4]
h_sez = np.linspace(0.001, 0.3, 500) # za non log scale plot
#h_sez = np.logspace(np.log10(3), np.log10(1e-5), 100) # za log scale plot

"""
# Initialize lists to store results for each step size
time_values = []
integration_values = []

# Perform integration for each step size and store the results
for h in h_sez:
    t, numerical_solution = integrate(euler, f, x0, h)
    integration_values.append(numerical_solution)
    time_values.append(t)
"""

# Function to perform calculations in a thread
def worker(h, time_results, int_results, index):
    ti, result = integrate(euler, f, x0, h)
    
    # Lock to ensure thread-safe access to the result_array
    with lock:
        #time_results[index] = ti
        #int_results[index] = result

        time_results.append(ti)
        int_results.append(result)

    #print(f"h {h}: Result {result}")
    print(f"h {h}")


# Array to store results in order
#time_results = np.zeros_like(h_sez)
#int_results = np.zeros_like(h_sez)
time_results = []
int_results = []
#err_results = [np.abs(int_results[i] - analyt(t, x0)) for i, t in enumerate(time_results)]



# Lock for thread-safe access to the results array
lock = threading.Lock()

# Create threads
threads = []
for i, h in enumerate(h_sez):
    thread = threading.Thread(target=worker, args=(h, time_results, int_results, i))
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


#plt.figure(figsize=(12, 6))
fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), ncols=2)

ax1.plot(t, x_analyt, ls='--', c='blue', alpha=0.8, lw=0.8, label='Analytical')
for i, h in enumerate(h_sez[::-1]):
    print(i, h)

    #plt.subplot( 1, 2, 1 )
    ax1.plot(time_results[i], int_results[i], ls='-', c=barve[-i])#, alpha=(1-h*0.1), lw=(1-h*0.1), label=f'h = {h}')
    #ax1.set_yscale('log')
    ax1.set_xlabel(r'time t [s]')
    ax1.set_ylabel(r'T [$^{\circ}$C]')
    ax1.set_title(r'Rezultati za različne velikosti koraka')
    #ax1.legend()

    xh_analyt = T_out + np.exp(-k*time_results[i])*(x0 - T_out)
    #plt.subplot( 1, 2, 2 )
    ax2.plot(time_results[i], np.abs(int_results[i] - analyt(time_results[i], x0)), ls='-', c=barve[i])#, alpha=(1-h*0.1), lw=(1-h*0.1), label=f'h = {h}')
    #ax2.set_yscale('log')
    ax2.set_xlabel(r'time t [s]')
    ax2.set_ylabel(r'$\Delta\varepsilon$')
    ax2.set_title(r'Absolutna napaka')
    #plt.legend()

ax1.legend()
    
# Add a colorbar
segments = [np.column_stack((x, y)) for x, y in zip(time_results, int_results)]
#print(segments)
col = LineCollection(segments, cmap="viridis_r", norm=color.Normalize())
col.set_array(h_sez)
ax1.add_collection(col, autolim=True)

# Create a colorbar with LogNorm on the second axis
cbar = plt.colorbar(col, ax=ax2, label="Velikost koraka h", norm=color.LogNorm())
cbar.set_label(r"Velikost koraka integracije - $h$")

plt.tight_layout()
plt.show()
