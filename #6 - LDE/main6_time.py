import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as color
import time
import psutil

from diffeq import *

from scipy.integrate import odeint

import threading


# Functions ____________________________________________________________________
def f( x, t, k=0.1, T_out=-5.0 ):
    return -k*(x - T_out)

def analyt( t, x0, k=0.1, T_out=-5.0 ):
    return  T_out + np.exp(-k*t)*(x0 - T_out)

def integrate(method, f, x0, h):
    n = int((b-a)/np.abs(h))
    t = np.linspace( a, b, n )

    if method == rk45:
        values, _ = method(f, x0, t)
    elif method == rkf:
        t, values = method(f, a, b, x0, 10, 0.3, 1e-6)
    elif method == odeint:
        results, _ = odeint(f, x0, t, full_output=1)
        values = results[:, 0]
    else:
        values = method(f, x0, t)
    return t, values


a, b = ( 0.0, 31.0 )

k = 0.1
T_out = -5.0
x0 = 21.0
t = np.linspace( a, b, 10 )
x_analyt = analyt(t, x0)

h_sez = [2.0, 1.0, 0.5, 0.1, 0.001, 1e-4]
#h_sez = np.linspace(1e-7, 0.1, 1000) # za non log scale plot
#h_sez = np.logspace(np.log10(0.3), np.log10(1e-5), 50) # za log scale plot
h_sez = np.logspace(np.log10(0.3), np.log10(1e-5), 50) # za log scale plot

#print(odeint(f, x0, t))
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

# Multithreading _______________________________________________________________
# Function to perform calculations in a thread
def worker(h, time_results, int_results, exec_results, err_results, index):
    global x0
    
    exec_vmesni = []
    err_vmesni = []
    """
    # Euler ____________________________________________________________
    start_time = time.process_time()
    ti, result = integrate(euler, f, x0, h)
    execution_time = time.process_time() - start_time  
    err = np.sum(np.abs(result - analyt(ti, x0)))/len(result)
    exec_vmesni.append(execution_time)
    err_vmesni.append(err)
    #time_results.append(ti)
    #int_results.append(result)

    # Heun _____________________________________________________________
    start_time = time.process_time()
    ti, result = integrate(heun, f, x0, h)
    execution_time = time.process_time() - start_time  
    err = np.sum(np.abs(result - analyt(ti, x0)))/len(result)
    exec_vmesni.append(execution_time)
    err_vmesni.append(err)
    #time_results.append(ti)
    #int_results.append(result)

    # Midpoint _________________________________________________________
    start_time = time.process_time()
    ti, result = integrate(rk2a, f, x0, h)
    execution_time = time.process_time() - start_time  
    err = np.sum(np.abs(result - analyt(ti, x0)))/len(result)
    exec_vmesni.append(execution_time)
    err_vmesni.append(err)
    #time_results.append(ti)
    #int_results.append(result)

    # Runge-Kutta 2 ____________________________________________________
    start_time = time.process_time()
    ti, result = integrate(rk2b, f, x0, h)
    execution_time = time.process_time() - start_time  
    err = np.sum(np.abs(result - analyt(ti, x0)))/len(result)
    exec_vmesni.append(execution_time)
    err_vmesni.append(err)
    #time_results.append(ti)
    #int_results.append(result)

    # Runge-Kutta 4 ____________________________________________________
    start_time = time.process_time()
    ti, result = integrate(rku4, f, x0, h)
    execution_time = time.process_time() - start_time  
    err = np.sum(np.abs(result - analyt(ti, x0)))/len(result)
    exec_vmesni.append(execution_time)
    err_vmesni.append(err)
    #time_results.append(ti)
    #int_results.append(result)
    """
    # Runge-Kutta 45 ___________________________________________________
    start_time = time.process_time()
    ti, result = integrate(rk45, f, x0, h)
    execution_time = time.process_time() - start_time  
    err = np.sum(np.abs(result - analyt(ti, x0)))/len(result)
    exec_vmesni.append(execution_time)
    err_vmesni.append(err)
    #time_results.append(ti)
    #int_results.append(result)

    
    # Runge-Kutta-Fehlberg _____________________________________________
    start_time = time.process_time()
    ti, result = integrate(rkf, f, x0, h)
    execution_time = time.process_time() - start_time  
    err = np.sum(np.abs(result - analyt(ti, x0)))/len(result)
    exec_vmesni.append(execution_time)
    err_vmesni.append(err)
    #time_results.append(ti)
    #int_results.append(result)
    

    # Predictor-corrector ______________________________________________
    start_time = time.process_time()
    ti, result = integrate(pc4, f, x0, h)
    execution_time = time.process_time() - start_time  
    err = np.sum(np.abs(result - analyt(ti, x0)))/len(result)
    exec_vmesni.append(execution_time)
    err_vmesni.append(err)
    #time_results.append(ti)
    #int_results.append(result)
    

    # Odeint ___________________________________________________________
    start_time = time.process_time()
    ti, result = integrate(odeint, f, x0, h)
    execution_time = time.process_time() - start_time  
    err = np.sum(np.abs(result - analyt(ti, x0)))/len(result)
    exec_vmesni.append(execution_time)
    err_vmesni.append(err)
    #time_results.append(ti)
    #int_results.append(result)


    exec_results.append(exec_vmesni)
    err_results.append(err_vmesni)
    
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
exec_results = []
err_results = []
#err_results = [np.abs(int_results[i] - analyt(t, x0)) for i, t in enumerate(time_results)]



# Lock for thread-safe access to the results array
lock = threading.Lock()

# Create threads
threads = []
for i, h in enumerate(h_sez):
    thread = threading.Thread(target=worker, args=(h, time_results, int_results, exec_results, err_results, i))
    threads.append(thread)

# Start the threads
for thread in threads:
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

# Print the final results array
#print("Final Results:", results)

# Plotting _____________________________________________________________________
cmap = plt.get_cmap('viridis')
cmap2 = plt.get_cmap('magma_r')
barve = [cmap(value) for value in np.linspace(0, 1, len(h_sez))]
barve2 = [cmap2(value) for value in np.linspace(0, 1, len(h_sez))]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#a93226']
#print("len: ", len(time_results), len(int_results), len(err_results))


"""
#plt.figure(figsize=(12, 6))
fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), ncols=2)

for i, h in enumerate(h_sez[::-1]):
    print(i, h)

    
    xh_analyt = T_out + np.exp(-k*time_results[i])*(x0 - T_out)
    #plt.subplot( 1, 2, 2 )
    ax1.plot(time_results[i], np.abs(int_results[i] - analyt(time_results[i], x0)), ls='-', c=barve[i])#, alpha=(1-h*0.1), lw=(1-h*0.1), label=f'h = {h}')
    #ax1.set_yscale('log')
    ax1.set_xlabel(r'time t [s]')
    ax1.set_ylabel(r'$\Delta\varepsilon$')
    ax1.set_title(r'Absolutna napaka')
    #plt.legend()

    #plt.subplot( 1, 2, 1 )
    ax2.plot(h_sez, exec_results[:, i], ls='-', c=barve[-i])#, alpha=(1-h*0.1), lw=(1-h*0.1), label=f'h = {h}')
    #ax2.set_yscale('log')
    ax2.set_xlabel(r'time t [s]')
    ax2.set_ylabel(r'T [$^{\circ}$C]')
    ax2.set_title(r'Rezultati za različne velikosti koraka')
    #plt.legend()


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
"""

methods = ['euler', 'heun', 'midpoint', 'Runge-Kutta 2', 'Runge-Kutta 4', 'Runge-Kutta 45', 'Runge-Kutta-Fehlberg', 'predictor-corrector', 'Odeint']
methods = ['euler', 'heun', 'midpoint', 'Runge-Kutta 2', 'Runge-Kutta 4', 'Odeint']
methods = ['Runge-Kutta 45', 'Runge-Kutta-Fehlberg', 'predictor-corrector', 'Odeint']

exec_results = np.array(exec_results)
err_results = np.array(err_results)
print(err_results)

plt.figure(figsize=(12, 6))

plt.subplot( 1, 2, 1 )
for i in range(len(exec_results[0])):
    print(i)
    plt.plot(h_sez, exec_results[:, i], ls='-', c=colors[i], label=f'{methods[i]}')
    #plt.plot(exec_results[:, i], h_sez, ls='-', c=colors[i])#, alpha=(1-h*0.1), lw=(1-h*0.1), label=f'h = {h}')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'Velikost koraka - $h$')
plt.ylabel(r'Čas izvajanja [$s$]')
plt.legend()

plt.subplot( 1, 2, 2 )
for i in range(len(exec_results[0])):
    print(i)
    plt.plot(h_sez, err_results[:, i], ls='-', c=colors[i], label=f'{methods[i]}')
    #plt.plot(exec_results[:, i], h_sez, ls='-', c=colors[i])#, alpha=(1-h*0.1), lw=(1-h*0.1), label=f'h = {h}')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'Velikost koraka - $h$')
plt.ylabel(r'Globalna absolutna napaka - $\Delta\varepsilon$')
plt.legend()


plt.tight_layout()
plt.show()