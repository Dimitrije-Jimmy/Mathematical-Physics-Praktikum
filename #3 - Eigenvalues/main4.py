import numpy as np
import matplotlib.pyplot as plt
import scipy.special

#import matplotlib.animation as animation
#import threading
import sys
import os

from numpy import linalg as LA

# Householder QR tridiagonal ___________________________________________________
def qr(M):
    A = np.copy(M)
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
    #for i in range(n):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A

def make_householder(a):
    #rescaling to v and sign for numerical stability
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.outer(v,v)
    return H


def zero_filter2(matrix, epsilon):
    A = np.copy(matrix)
    A[A < epsilon] = 0.0
    return A


# Quantum Mechanics ____________________________________________________________
def lho(n):
    return np.diag([i + 1/2 for i in range(0, n)])

def delta(i, j):
    return int(i == j)

def q_matrix_single(n):
    matrix = np.zeros((n, n))
    for ind, val in np.ndenumerate(matrix):
        i, j = ind
        matrix[i][j] = 0.5 * np.sqrt(i + j + 1) * delta(np.abs(i - j), 1)

    return np.matmul(matrix, np.matmul(matrix, np.matmul(matrix, matrix)))

def basis(q, n):
    return (2**n * np.math.factorial(n) * np.sqrt(np.pi))**(-1/2) * np.exp(-q**2/2) * scipy.special.eval_hermite(n, q)

def anharmonic(lamb, n, func):
    return lho(n) + lamb*func(n)


def plot_poly(x, vect):
    c = [vect[i]*basis(x, i) for i in range(len(vect))]
    return np.sum(c)


# Eigenvalue function __________________________________________________________
def diagonalize(matrix, tol=10**-15, maxiter=1000):
    """
    A diagonalization and eigenvector calculation method
    :param matrix: input matrix to diagonalize
    :param tol: tolerance for zero filter
    :param maxiter: maximum number of iterations
    :return Diagonalized matrix and matrix of eigenvectors
    """
    tridiag = matrix
    #tridiag, _ = qr(matrix)
    s = np.eye(tridiag.shape[0])  # To store eigenvectors
    for i in range(maxiter):
        print(f"Iter: {i}")
        if i == maxiter - 1:
            raise Warning("Maximum number of iterations exceeded")
        test =  np.abs(np.sum(np.abs(tridiag)) - np.sum(np.abs(np.diag(tridiag))))
        if test < 10**-10:
            print(i)
            break

        Q, R = qr(tridiag)
        tridiag = zero_filter2(np.matmul(Q.T, np.matmul(tridiag, Q)), tol)
        
        #tridiag[np.matmul(Q.T, np.matmul(tridiag, Q)) < tol] = 0.0
        s = np.matmul(s, Q)

    return tridiag, s


lam = 0.5
data = anharmonic(lam, 10, q_matrix_single)
# data = anharmonic(lam, 10, q_matrix_double)
# data = anharmonic(lam, 10, q_matrix_quad)

diag, Q = diagonalize(data, tol=10**-5, maxiter=1000)


# Time, RAM, CPU _______________________________________________________________
import time
import psutil
from memory_profiler import memory_usage
from functools import partial


"""
def benchmark_eigenvalue_function(eigenvalue_function, matrix, num_repeats=2):
    execution_times = []
    memory_usages = []
    cpu_percentages = []

    for _ in range(num_repeats):
        start_time = time.time()
        # Measure memory usage
        memory_usage_start = memory_usage((eigenvalue_function, (matrix,)))

        # Execute the eigenvalue function
        eigenvalue_function(matrix)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Measure memory usage again
        memory_usage_end = memory_usage((eigenvalue_function, (matrix,)))
        max_memory_usage = max(memory_usage_end) - max(memory_usage_start)

        # Measure CPU percentage
        cpu_percentage = psutil.cpu_percent()

        execution_times.append(execution_time)
        memory_usages.append(max_memory_usage)
        cpu_percentages.append(cpu_percentage)

    avg_execution_time = sum(execution_times) / num_repeats
    avg_memory_usage = sum(memory_usages) / num_repeats
    avg_cpu_percentage = sum(cpu_percentages) / num_repeats

    return {
        'avg_execution_time': avg_execution_time,
        'avg_memory_usage': avg_memory_usage,
        'avg_cpu_percentage': avg_cpu_percentage
    }
"""

def measure_function_performance(func, matrix):
    start_time = time.time()
    
    # Execute the eigenvalue function
    func(matrix)

    execution_time = time.time() - start_time

    return execution_time
"""
def benchmark_eigenvalue_function(eigenvalue_function, matrix):
    # Ensure the function takes only one argument
    func = partial(eigenvalue_function, matrix)

    memory_usage_start = memory_usage((psutil.Process().memory_info().rss,))
    cpu_percent_start = psutil.cpu_percent()

    start_time = time.time()

    # Execute the eigenvalue function
    func(matrix)

    execution_time = time.time() - start_time

    memory_usage_end = memory_usage((psutil.Process().memory_info().rss,))
    cpu_percent_end = psutil.cpu_percent()

    max_memory_usage = max(memory_usage_end) - max(memory_usage_start)
    avg_cpu_percentage = cpu_percent_end - cpu_percent_start


    avg_execution_time = sum(execution_times) / num_repeats
    avg_memory_usage = sum(memory_usages) / num_repeats
    avg_cpu_percentage = sum(cpu_percentages) / num_repeats

    return {
        'avg_execution_time': avg_execution_time,
        'avg_memory_usage': avg_memory_usage,
        'avg_cpu_percentage': avg_cpu_percentage
    }

matrix = anharmonic(lam, 10, q_matrix_single)  # Your NxN matrix
result = benchmark_eigenvalue_function(diagonalize, matrix)

print("Average Execution Time: {:.4f} seconds".format(result['avg_execution_time']))
print("Average Memory Usage: {:.4f} MiB".format(result['avg_memory_usage']))
print("Average CPU Usage: {:.4f}%".format(result['avg_cpu_percentage']))
"""

matrix = anharmonic(lam, 10, q_matrix_single)  # Your NxN matrix
result = measure_function_performance(diagonalize, matrix)
print(result)