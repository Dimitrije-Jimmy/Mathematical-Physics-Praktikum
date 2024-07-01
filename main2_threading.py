import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scs
from scipy.sparse.linalg import spsolve
from scipy.sparse import block_diag

import threading
import time
import csv
import os
dir_path = os.path.dirname(os.path.realpath(__file__))+'\\Images\\Results\\'
print(dir_path)

from main1_functions import *

# Functions ____________________________________________________________________

def galerkin_vectorized(M_max, N_max):
    m_values, n_values = np.meshgrid(range(M_max + 1), range(1, N_max + 1), indexing='ij')
    
    b_values = np.zeros(((M_max + 1) * N_max))
    b_values[np.arange(len(b_values))] = bj_vectorized(m_values.flatten(), N_max).flatten()

    Aji = [Aij_vectorized(m_values[i, j], N_max) for i, j in np.ndindex(m_values.shape)]
    A = block_diag(Aji, format='csr') 
    a_values = spsolve(A, b_values)
    C_values = -(32./np.pi) * np.dot(b_values, a_values)

    return C_values, a_values


def bj_vectorized(m_values, N_max):
    j_values = np.arange(N_max) + 1
    beta_values = scs.beta(2 * m_values + 3, j_values + 1)
    b_values = -2 * beta_values / (2 * m_values + 1)
    return b_values.reshape((len(m_values), N_max))


def Aij_vectorized(m_values, N_max):
    i_values, j_values = np.meshgrid(range(N_max), range(N_max), indexing='ij')
    nn_values = i_values
    n_values = j_values
    beta_values = scs.beta(n_values + nn_values - 1, 3 + 4 * m_values)
    A_values = -beta_values * 0.5 * np.pi * n_values * nn_values * (3 + 4 * m_values) / (2 + 4 * m_values + n_values + nn_values)
    return A_values


def err_calc_threaded(lock, results, M, N, C100, a100):
    start_time = time.process_time()
    #C_values, a_values = galerkin_vectorized(M, N)
    C_values, a_values = galerkin(M, N)
    execution_time = start_time - time.process_time()
    err_C_values = np.abs(C_values - C100)
    #err_a_values = np.abs(a_values - a100)

    with lock:
        results.append((M, N, C_values, a_values, err_C_values, execution_time))


def err_time(M_max_sez, N_max_sez, C100, a100):
    # Lock for thread-safe access to the results array
    lock = threading.Lock()

    print("Starting")

    # Create threads and results list
    threads = []
    results = []

    for M in M_max_sez:
        for N in N_max_sez:
            print(M, N)
            thread = threading.Thread(target=err_calc_threaded, args=(lock, results, M, N, C100, a100))
            threads.append(thread)

    # Start the threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print("Done")
    return results

from io import StringIO
def write_results_to_csv(filename, results):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['M', 'N', 'C', 'a', 'err_C', 'execution_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            # Convert array to string using np.savetxt
            #array_string = np.array2string(result[3], separator=',').replace('\n', '')[1:-1]
            
            # Use StringIO to write the array to a string buffer
            buffer = StringIO()
            np.savetxt(buffer, result[3])
            array_string = buffer.getvalue().strip()
            
            writer.writerow({
                'M': result[0],
                'N': result[1],
                'C': result[2],
                #'a': str(result[3]),  # Convert array to string
                'a': array_string,
                'err_C': result[4],
                'execution_time': result[5]
            })
            
def write_results_to_csv_brez_a(filename, results):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['M', 'N', 'C', 'a', 'err_C', 'execution_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow({
                'M': result[0],
                'N': result[1],
                'C': result[2],
                'err_C': result[4],
                'execution_time': result[5]
            })


# Execution ____________________________________________________________________

# Usage example:
# results = ...  # obtain results from threaded execution

C100, a100 = galerkin(100, 100)
print(C100)
print(a100)

mmax = np.array([0,1,2,5,10,20,30,50,75,100])
nmax = np.array([0,1,2,5,10,20,30,50,75,100])
#nmax = (100 * np.ones(len(mmax))).astype(int)
mmax = np.array([100])
nmax = np.array([100])

rezultati =  err_time(mmax, nmax, C100, a100)
rezultati3 =  err_time([3], [3], C100, a100)
rezultati5 =  err_time([5], [5], C100, a100)

#write_results_to_csv(dir_path+'output.csv', rezultati)
#write_results_to_csv(dir_path+'C100a100.csv', rezultati)
write_results_to_csv(dir_path+'C3a3_threaded.csv', rezultati3)
write_results_to_csv(dir_path+'C5a5_threaded.csv', rezultati5)
#write_results_to_csv_brez_a(dir_path+'output_brez_a.csv', rezultati)