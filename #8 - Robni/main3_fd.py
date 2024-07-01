import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from diffeq import *
from bvp import *


# Functions ____________________________________________________________________

def analytic(x, k, A=5.0):
    
    if k % 2 == 0:
        return np.sin(k/(2*A)*np.pi*x)  
    else:
        return np.cos(k/(2*A)*np.pi*x)

def analytic_finite2(x, k, A=5.0):
    
    if np.abs(x) <= 1:
        if k % 2 == 0:
            return np.sin(k/(2*A)*np.pi*x)  
        else:
            return np.cos(k/(2*A)*np.pi*x)
    else:
        return np.exp(-a*np.abs(x))
    
"""
def analytic_finite(x, k, A=5.0):
    ind = np.where(np.abs(x) <= 1)[0]
    a, b = ind[0], ind[-1]
    
    vec1 = np.exp(-a*np.abs(x[:a]))
    vec3 = np.exp(-a*np.abs(x[b:]))
    if k % 2 == 0:
        vec2 = np.sin(k/(2*A)*np.pi*x[a:b])  
    else:
        vec2 = np.cos(k/(2*A)*np.pi*x[a:b])
        
    return np.concatenate(np.concatenate(vec1, vec2), vec3)
    
    return np.array(vec1 + vec2 + vec3)
"""

def analytic_finite3(x, k, A=5.0):
    result = np.exp(-A*np.abs(x))
    mask = np.abs(x) <= 1
    result[mask] = np.sin(k/(2*A)*np.pi*x[mask]) if k % 2 == 0 else np.cos(k/(2*A)*np.pi*x[mask])
    return result
    
def analytic_finite4(x, k, A=5.0):
    result = np.empty_like(x)
    
    condition = np.abs(x) <= 1
    result[condition] = np.where(k % 2 == 0,
                                 np.sin(k/(2*A)*np.pi*x[condition]),
                                 np.cos(k/(2*A)*np.pi*x[condition]))
    
    result[~condition] = np.exp(-A*np.abs(x[~condition]))
    
    return result


def analytic_finite(x, k, A=5.0):
    result = np.empty_like(x)

    for i, val in enumerate(x):
        print(i, val)
        if np.abs(val) <= 1.0:
            if k % 2 == 0:
                result[i] = np.sin(k / (2 * A) * np.pi * val)
            else:
                result[i] = np.cos(k / (2 * A) * np.pi * val)
        else:
            result[i] = np.exp(-A * np.abs(val))

    return result


def matrika_neskoncen(n):
    A = np.zeros((n, n))
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:, 1:], 1)
    np.fill_diagonal(A, -2)
    A[0, 1] = 1
    A[-1, -2] = 1

    return A


def fd_ipw(n, a, V):
    x = np.linspace(-a, a, n)
    h = x[1] - x[0]
    V = np.zeros(n)
    diag = 1/h**2 * matrika_neskoncen(n)
    H = -1/2 * diag + np.diag(V)
    eigE, eigPsi = np.linalg.eigh(H)

    return x, eigE, eigPsi/np.max(eigPsi)


def fd_fpw(n, a, V):
    x = np.linspace(-a, a, n)
    h = x[1] - x[0]
    dim = len(x)
    pos = int(dim // 2.2)
    width = int(2 * (dim / 2 - pos))
    V_fpw = np.zeros(dim)
    V_fpw[:pos] = V
    V_fpw[(pos + width):] = V
    diag = 1/h**2 * matrika_neskoncen(n)
    H = -1/2 * diag + np.diag(V_fpw)
    eigE, eigPsi = np.linalg.eigh(H)

    return x, eigE, eigPsi/np.max(eigPsi), V_fpw
    #return x, eigE, eigPsi*dim/a, V_fpw


# Set initial conditions and parameters
V = 100.0
a = 10.0


# Plotting _____________________________________________________________________


# Neskončna jama _______________________________________________________

# FD method IPW plot
dim = 1000
x, eigE, eigPsi = fd_ipw(dim, a, 0)
print(eigE[:10])
#x, eigE, eigPsi, V_fpv = fd_fpw(dim, a, 0)
#print(eigE[:10])
"""
plt.plot(x, eigPsi[:, 0]/ (np.sqrt(np.sum(eigPsi[:, 0]**2)*(20)/len(eigPsi[:, 0]))), label=f"$\\psi_1$, $E_1= {np.around(eigE[1], 2)}$")
plt.plot(x, eigPsi[:, 1]**2, label=f"$\\psi_1$, $E_1= {np.around(eigE[2], 2)}$")
plt.plot(x, analytic(x, 1, a)**2, label=f"analiticna 1$")
plt.plot(x, analytic(x, 2, a)**2, label=f"analiticna 2$")

plt.legend()
plt.show()
"""
#sys.exit()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

stvr = 5

cmap = plt.get_cmap('viridis')
cmap2 = plt.get_cmap('magma_r')
barve = [cmap(value) for value in np.linspace(0, 1, stvr)]
for i in range(stvr):
    if i % 2 == 0:
        ax1.plot(x, -eigPsi[:, i], c=barve[i], label=f"$\\psi_{i+1}$, $E_{i+1}= {np.around(eigE[i]/eigE[0], 3)}E_1$")
        err = np.abs(-eigPsi[:, i] - analytic(x, i+1, a))
    else:
        ax1.plot(x, eigPsi[:, i], c=barve[i], label=f"$\\psi_{i+1}$, $E_{i+1}= {np.around(eigE[i]/eigE[0], 3)}E_1$")
        err = np.abs(eigPsi[:, i] - analytic(x, i+1, a))
    #ax1.plot(x, analytic(x, i+1, a), label=f"analiticna {i}$")

    ax2.plot(x, err, c=barve[i], label=f"$\\psi_{i+1}$")
    
    
ax1.set_title("Neskočna potencialna jama")
ax1.set_xlabel("x")
ax1.set_ylabel(r"$\psi_n(x)$")
ax1.legend(loc="upper right")


ax2.set_title(f"Absolutne napake rešitev in $n = {dim}$")
ax2.set_xlabel("x")
ax2.set_ylabel(r"$\Delta\psi_n(x)$")
ax2.set_yscale("log")
ax2.legend()

plt.tight_layout()
plt.show()
plt.clf()
plt.close()


fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
#fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))

stvr = 5
for i in range(stvr):
    if i % 2 == 0:
        ax1.plot(x, eigPsi[:, i]**2, c=barve[i], label=f"$\\psi_{i+1}$, $E_{i+1}= {np.around(eigE[i]/eigE[0], 3)}E_1$")
        err = np.abs(eigPsi[:, i]**2 - analytic(x, i+1, a)**2)
    else:
        ax1.plot(x, eigPsi[:, i]**2, c=barve[i], label=f"$\\psi_{i+1}$, $E_{i+1}= {np.around(eigE[i]/eigE[0], 3)}E_1$")
        err = np.abs(eigPsi[:, i]**2 - analytic(x, i+1, a)**2)
    #ax1.plot(x, analytic(x, i+1, a), label=f"analiticna {i}$")

    ax2.plot(x, err, c=barve[i], label=f"$|\\psi_{i+1}|^2$")
    
    
ax1.set_title("Neskočna potencialna jama")
ax1.set_xlabel("x")
ax1.set_ylabel(r"$|\psi_n(x)|^2$")
ax1.legend(loc="upper right")


ax2.set_title(f"Absolutne napake rešitev in $n = {dim}$")
ax2.set_xlabel("x")
ax2.set_ylabel(r"$\Delta|\psi_n(x)|^2$")
ax2.set_yscale("log")
ax2.legend()

plt.tight_layout()
plt.show()
plt.clf()
plt.close()

# Končna jama __________________________________________________________


# FD method FPW plot
dim = 1000
x_fpw, eigE_fpw, eigPsi_fpw, V_fpw = fd_fpw(dim, a, V)
n_solutions = np.shape(eigE_fpw)[0]
print(f"Found {n_solutions} solutions!")
stvr = 5

#print(eigE[:10])

fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))

amplitude = 5
#ax1.plot(x_fpw, V_fpw, 'k--', lw=1, alpha=0.8)
ax1.plot([-a, -1], [V, V], c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([1, a], [V, V], c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([-1, 1], [0, 0], c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([-1, -1], [0, V], c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([1, 1], [0, V], c='k', ls='--', lw=0.8, alpha=0.8)
for i in range(stvr):
    #ax1.plot(x_fpw, 50 * eigPsi_fpw[:, i], label=f"$\\psi_1$, $E_1={np.around(eigE_fpw[i], 2)}$")
    vektorji = amplitude*eigPsi_fpw[:, i] + i*V/stvr
    ax1.plot(x_fpw, vektorji, label=f"$\\psi_{i+1}$, $E_{i+1}={np.around(eigE_fpw[i]/eigE_fpw[0], 3)}E_1$")
    
    ax1.text(1.35, (i+0.1)*V/stvr, f'$\\psi_{i+1}$, $E_{i+1} = {np.around(eigE_fpw[i]/eigE_fpw[0], 3)}E_1$', verticalalignment='center')
    
    #ax1.plot(x_fpw, analytic_finite(x_fpw, i+1, a), label=f"$\\psi_{i}$, analytic")

    # Fill the area under the curve with a solid color (e.g., light blue)
    ax1.fill_between(x_fpw, vektorji, y2=i*V/stvr, alpha=0.5)
    
    # absolutne napake ni ker nimamo analiticne resitve...
    #err = np.abs(eigPsi_fpw[:, i] - analytic(x_fpw, i+1, a))
    #ax2.plot(x_fpw, err, label=f"$\\psi_{i+1}$")

ax1.set_title(f"Končna jama globine {V}")
ax1.set_xlabel("x")
ax1.set_ylabel(r"$\psi_n(x)$")
#ax1.legend(loc="upper right")
ax1.set_xlim(-2, 2)

#ax2.set_title(f"Absolutne napake rešitev z FD in $n = {dim}$")
#ax2.set_xlabel("x")
#ax2.set_ylabel(r"$|num - analytic|$")
#ax2.set_yscale("log")
#ax2.legend()

plt.tight_layout()
plt.show()
plt.clf()
plt.close()


# natančnost ___________________________________________________________
import time


dimensions = np.arange(100, 1000, 200)


x_val = []
psi_val = []
E_val = []
psi_err_val = []

for dim in dimensions:
    start_time = time.process_time()
    x, E, Psi = fd_ipw(dim, a, 0)    
    execution_time = time.process_time() - start_time  
    err_vmes = []
    for i in range(10):
        if i % 2 == 0:
            #ax1.plot(x, Psi[:, i], c=barve[i], label=f"$\\psi_{i+1}$, $E_{i+1}= {np.around(E[i]/E[0], 3)}E_1$")
            err = np.abs(-Psi[:, i] - analytic(x, i+1, a))
        else:
            #ax1.plot(x, Psi[:, i], c=barve[i], label=f"$\\psi_{i+1}$, $E_{i+1}= {np.around(E[i]/E[0], 3)}E_1$")
            err = np.abs(Psi[:, i] - analytic(x, i+1, a))
        err_vmes.append(err)
        
    x_val.append(x)
    psi_val.append(Psi)
    E_val.append(E)
    psi_err_val.append(err_vmes)


plt.plot(x_val)

        
        
                        
    
    




sys.exit()
# Čas izvajanja ________________________________________________________________
import time

h_sez = [2.0, 1.0, 0.5, 0.1, 0.001]#, 1e-4]
#h_sez = np.linspace(0.001, 0.3, 500) # za non log scale plot
#h_sez = np.logspace(np.log10(3), np.log10(1e-5), 100) # za log scale plot


def worker(h, time_results, x_results, exec_results, 
           xerr_results, index):
    
    x_vmesni = np.zeros_like(methods, dtype=object)
    time_vmesni = np.zeros_like(methods, dtype=object)
    xerr_vmesni = np.zeros_like(methods, dtype=object)
    exec_vmesni = np.zeros_like(methods, dtype=object)
    #ram_vmesni = np.zeros_like(methods, dtype=object)

    for i, metoda in enumerate(methods):
        start_time = time.process_time()
        ti, x, v = integrate(metoda, funkcija, p0, h)
        
        execution_time = time.process_time() - start_time  
        
        xerr = np.abs(-eigPsi[:, i] - analytic(x, i+1, a))
        xerr = np.sum(np.abs(x - exact(ti, x0, w0)))/len(x)
        
        time_vmesni[i] = ti
        x_vmesni[i] = x
        xerr_vmesni[i] = xerr
        exec_vmesni[i] = execution_time
        
    
    # Lock to ensure thread-safe access to the result_array
    with lock:
        #print(exec_vmesni)
        
        time_results[index] = time_vmesni 
        x_results[index] = x_vmesni 
        exec_results[index] = exec_vmesni 
        xerr_results[index] = xerr_vmesni 


    #print(f"h {h}: Result {result}")
    print(f"h {h}")


# Array to store results in order
time_results = np.zeros_like(h_sez, dtype=object)
x_results = np.zeros_like(h_sez, dtype=object)
exec_results = np.zeros_like(h_sez, dtype=object)
xerr_results = np.zeros_like(h_sez, dtype=object)



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

time_results = np.stack(time_results, dtype=numpy.dtype)
x_results = np.stack(x_results, dtype=numpy.dtype)
exec_results = np.stack(exec_results, dtype=numpy.dtype)
xerr_results = np.stack(xerr_results, dtype=numpy.dtype)


# Plotting _____________________________________________________________________
cmap = plt.get_cmap('viridis')
cmap2 = plt.get_cmap('magma_r')
barve = [cmap(value) for value in np.linspace(0, 1, len(h_sez))]
barve2 = [cmap2(value) for value in np.linspace(0, 1, len(h_sez))]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#a93226']

plt.figure(figsize=(15, 6))

plt.subplot( 1, 2, 1 )
for i in range(len(exec_results[0])):
    print(i)
    plt.scatter(h_sez, exec_results[:, i], c=colors[i], label=f'')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'Velikost koraka - $h$')
plt.ylabel(r'Čas izvajanja [$s$]')
plt.title(r'Čas izvajanja')
plt.legend()

plt.subplot( 1, 2, 2 )
plt.axhline(y=1e-3, ls='--', lw=0.8, c='k')
for i in range(len(exec_results[0])):
    print(i)
    plt.scatter(h_sez, xerr_results[:, i], c=colors[i], label=f'')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'Velikost koraka - $h$')
plt.ylabel(r'Globalna absolutna napaka - $\Delta\varepsilon$')
plt.title(r'Absolutna napaka')
plt.legend()

plt.tight_layout()


# Absolutna napaka v odvisnosti od execution časa + step size
fig, (ax1) = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)

cmap1 = plt.get_cmap('viridis')
cmap2 = plt.get_cmap('plasma')
cmap3 = plt.get_cmap('RdYlBu')
cmap4 = plt.get_cmap('winter')
cmap5 = plt.get_cmap('twilight')
colormaps = [cmap1, cmap2, cmap3, cmap4, cmap5]
for i in range(len(methods)):
    print(i)
    
    #ax1.plot(exec_results[:, i], xerr_results[:, i], ls='-', lw=1.0, alpha=0.5, label=f'{methods_names[i]}')
    q = ax1.scatter(exec_results[:, i], xerr_results[:, i], c=h_sez, cmap=colormaps[i], label=f'{methods_names[i]}')
    #ax1.plot(xerr_results[:, i], exec_results[:, i], ls='-', lw=1.0, alpha=0.9, label=f'{methods_names[i]}')
    #q = ax1.scatter(xerr_results[:, i], exec_results[:, i], c=h_sez, cmap=colormaps[i])
    
    # Add colorbar for reference
    #cbar = plt.colorbar(q, shrink=0.6, pad=0.05)
    #cbar.set_label(f'{methods_names[i]}')


ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'Čas izvajanja [$s$]')
ax1.set_ylabel(r'Absolutna napaka')
ax1.set_title(r'Željena natančnost v odvisnosti od časa izvajanja')
ax1.legend()

plt.tight_layout()

plt.show()
