import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import time
import threading

from main1_functions import *

dir_path = os.path.dirname(os.path.realpath(__file__))+'\\Results\\'
print(dir_path)

# Initialization _______________________________________________________________

# prvi del _____________________________________________________________
k = 1.
w = k**0.5

w = 0.2
k = w**2.
lamb = 10.
alpha = k**(1/4)

a = -40.
b = 40.
t0 = 0.
tnih = 2*np.pi / w
tf = 314.
tf = 5.*tnih


N = 300
M = 1000

dx = (b - a)/(N-1)
dt = (tf - t0)/(M-1)

x = np.linspace(a, b, N)
t = np.linspace(t0, tf, M)
#t = np.arange(t0, tf, dt)


# drugi del ____________________________________________________________
sigma0 = 1./20.
k0 = 50.*np.pi
lamb2 = 0.25

a2 = -0.5
b2 = 1.5
t02 = 0.
tf2 = 1.
tf2 = .005

dx2 = (b2 - a2)/(N-1)
dt = 2.*dx**2
x2 = np.arange(a2, b2+dx2, dx2)
t2 = np.arange(t02, tf2, dt)

x2 = np.linspace(a2, b2, N)
t2 = np.linspace(t02, tf2, M)



# ... (previous functions and initialization parameters)

# Function to calculate error and execution time
def calculate_error_execution_time(n):
    #for n in L:
    print(n)
    # Set N, M, and other parameters
    N = M = n
    N = 300

    x = np.linspace(a, b, N)
    t = np.linspace(t0, tf, M)

    x2 = np.linspace(a2, b2, N)
    t2 = np.linspace(t02, tf2, M)

    # prvi del
    start_time = time.process_time()
    phi1_anal = analytic_koherent2(x, t)
    zp1 = valovna_prvi(x)
    phi1 = C_N(x, t, zp1)
    elapsed_time_phi1 = time.process_time() - start_time

    napaka1 = np.abs(phi1 - phi1_anal)
    napaka1sq = np.abs(np.abs(phi1**2) - np.abs(phi1_anal**2))

    # drugi del
    start_time = time.process_time()
    phi2_anal = analytic_empty2(x, t)
    zp2 = valovna_drugi(x)
    phi2 = C_N(x, t, zp2)
    start_time = time.process_time()

    napaka2 = np.abs(phi2 - phi2_anal)
    napaka2sq = np.abs(np.abs(phi2**2) - np.abs(phi2_anal**2))

    # Save results to file using numpy
    np.savez(dir_path+f"results_N_{N}_M_{M}.npz",
                N=N,
                M=M,
                napaka1=napaka1,
                napaka1sq=napaka1sq,
                elapsed_time_phi1=elapsed_time_phi1,
                napaka2=napaka2,
                napaka2sq=napaka2sq,
                elapsed_time_phi2=elapsed_time_phi2)

# Example of calling the function with L values from 100 to 10000
#L_values = np.linspace(100, 2000, 50, dtype=int)
L_values = np.arange(100, 2100, 40, dtype=int)


# Lock for thread-safe access to the results array
lock = threading.Lock()

# Create threads
threads = []
for i, n in enumerate(L_values):
    thread = threading.Thread(target=calculate_error_execution_time, args=(n, ))
    threads.append(thread)

# Start the threads
for thread in threads:
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()
