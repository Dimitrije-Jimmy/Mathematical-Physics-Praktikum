import numpy as np
import matplotlib.pyplot as plt
import scipy.special

import os

from numpy import linalg as LA

# Householder QR tridiagonal ___________________________________________________
def qr_householder(M):
    A = np.copy(M)
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
    #for i in range(n):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)

    eigenvalues = np.diag(A)

    # Sort eigenvalues and corresponding eigenvectors in descending order of absolute values
    sort_order = np.argsort(np.abs(eigenvalues))[::-1]  # Reverse the order
    eigenvalues = eigenvalues[sort_order]
    Q = Q[:, sort_order]

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

def data_sort(diag, Q):
    diag_elements = np.diag(diag)
    vectors = np.copy(Q)
    output = []
    for coord, val in np.ndenumerate(diag_elements):
        output.append([val, vectors[coord[0]]])
    output.sort()

    return np.array(output)

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
    tridiag = matrix
    
    s = np.eye(tridiag.shape[0])  # To store eigenvectors

    for i in range(maxiter):
        #print(f"Iter: {i}")
        if i == maxiter - 1:
            raise Warning("Maximum number of iterations exceeded")
        test =  np.abs(np.sum(np.abs(tridiag)) - np.sum(np.abs(np.diag(tridiag))))
        if test < 10**-10:
            print(i)
            break

        Q, R = qr_householder(tridiag)
        tridiag = zero_filter2(np.matmul(Q.T, np.matmul(tridiag, Q)), tol)
        
        s = np.matmul(s, Q)

    eigenvalues = np.diag(tridiag)

    return eigenvalues, s


# Jacobi _______________________________________________________________________
def jacobi_eigenvalue(A, tol=1e-9, max_iter=1000):
    n = A.shape[0]
    Q = np.eye(n)  # Initialize the orthogonal transformation matrix
    eigenvalues = np.zeros(n)
    
    for _ in range(max_iter):
        # Find the indices (p, q) of the maximum off-diagonal element
        p, q = np.unravel_index(np.argmax(np.abs(A - np.diag(np.diag(A))), axis=None), A.shape)
        
        if abs(A[p, q]) < tol:
            break  # Convergence criteria met
        
        # Calculate the rotation angle theta
        if A[p, p] == A[q, q]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan(2 * A[p, q] / (A[p, p] - A[q, q]))
        
        # Construct the rotation matrix J
        J = np.eye(n)
        J[p, p] = J[q, q] = np.cos(theta)
        J[p, q] = -np.sin(theta)
        J[q, p] = np.sin(theta)
        
        # Update A and Q with the rotation
        A = np.dot(J.T, np.dot(A, J))
        Q = np.dot(Q, J)
    
    eigenvalues = np.diag(A)

    # Sort eigenvalues and corresponding eigenvectors in descending order of absolute values
    sort_order = np.argsort(np.abs(eigenvalues))[::-1]  # Reverse the order
    eigenvalues = eigenvalues[sort_order]
    Q = Q[:, sort_order]
    
    return eigenvalues, Q


def qr_algorithm_numpy(A, iterations=50):
    for i in range(iterations):
        # QR decomposition
        Q, R = LA.qr(A)
        A = np.dot(R, Q)

    eigenvalues = np.diag(A)
    return eigenvalues, Q



# Time threading _______________________________________________________________
from queue import Queue
import threading
import time


queue = Queue()
def fill_queue(n):
    for primer in n:
        queue.put(primer)

"""
def worker(list_of_functions, lamb):
    seznam_casov = []
    seznam_razlik = []
    
    while not queue.empty():
        global q_matrix_single

        matrix_size = queue.get()
        print("matrix size: ", matrix_size)

        data = anharmonic(lamb, matrix_size, q_matrix_single)

        time_list = []
        eigenvalues_list = []
        for function in list_of_functions:
            
            start_time = time.time()

            # Because LA.eig outputs complex values
            if function == LA.eig:
                complex_values, _ = function(data)
                eigenvalues = np.real(complex_values)
            # Execute the eigenvalue function
            eigenvalues, _ = function(data)

            execution_time = time.time() - start_time  

            time_list.append(execution_time)
            eigenvalues_list.append(eigenvalues) 
        
        print(eigenvalues_list)
        data_array = np.array(eigenvalues_list)
        # Calculate the element-wise absolute differences with the last sublist
        differences = np.abs(data_array - data_array[-1])
        # Sum the absolute differences along the rows
        razlika_lastnih_vrednosti = np.sum(differences, axis=1)

        
        #file.write(f"{matrix_size},\t {x_sez},\t {y_sez},\t {time1},\t {time2},\t {theta},\t {l},\t {razdalja1}\n")
        #file.flush()

        #matrix.append([x_sez, y_sez, time1, time2, theta, l, razdalja1])

        seznam_casov.append(n[t-1], time_list)
        seznam_razlik.append(n[t-1], razlika_lastnih_vrednosti)

    return seznam_casov, seznam_razlik
"""

#sezsez_casov = []
#sezsez_razlik = []
seznam_casov = []
seznam_razlik = []
def worker(list_of_functions, lamb):
    #seznam_casov = []
    #seznam_razlik = []
    
    while not queue.empty():
        global q_matrix_single

        matrix_size = queue.get()
        print("matrix size: ", matrix_size)

        data = anharmonic(lamb, matrix_size, q_matrix_single)

        time_list = []
        eigenvalues_list = []
        
        # Diagonalize
        start_time = time.time()
        eigenvalues, _ = diagonalize(data)
        print(eigenvalues)
        execution_time = time.time() - start_time  
        time_list.append(execution_time)
        eigenvalues_list += eigenvalues 

        # Jacobi
        start_time = time.time()
        eigenvalues, _ = jacobi_eigenvalue(data)
        execution_time = time.time() - start_time  
        time_list.append(execution_time)
        eigenvalues_list += eigenvalues 
        
        # QR numpy
        start_time = time.time()
        eigenvalues, _ = qr_algorithm_numpy(data)
        execution_time = time.time() - start_time  
        time_list.append(execution_time)
        eigenvalues_list += eigenvalues 

        # LA.eigh
        start_time = time.time()
        eigenvalues, _ = LA.eigh(data)
        execution_time = time.time() - start_time  
        time_list.append(execution_time)
        eigenvalues_list += eigenvalues 

        # LA.eig
        start_time = time.time()
        complex_values, _ = LA.eig(data)
        eigenvalues = np.real(complex_values)
        execution_time = time.time() - start_time  
        time_list.append(execution_time)
        eigenvalues_list += eigenvalues 
        
        print(time_list)
        #print(eigenvalues_list)
        data_array = np.array(eigenvalues_list)
        # Calculate the element-wise absolute differences with the last sublist
        differences = np.abs(data_array - data_array[-1])
        # Sum the absolute differences along the rows
        razlika_lastnih_vrednosti = np.sum(differences, axis=1)

        
        #file.write(f"{matrix_size},\t {x_sez},\t {y_sez},\t {time1},\t {time2},\t {theta},\t {l},\t {razdalja1}\n")
        #file.flush()

        #matrix.append([x_sez, y_sez, time1, time2, theta, l, razdalja1])

        seznam_casov.append([matrix_size, time_list])
        seznam_razlik.append([matrix_size, razlika_lastnih_vrednosti])

    #print(seznam_casov)
    #sezsez_casov.append(seznam_casov)
    #sezsez_razlik.append(seznam_razlik)

    #return seznam_casov, seznam_razlik


def run_threads(threads, n, lamb, eigenvalue_function):

    #file.write(f"N = {N}, ,\t mu = {mu},\t nu = {nu}\n")

    print("running")

    fill_queue(n)

    # Create a results queue to collect results
    results_queue = Queue()

    thread_list = []

    for t in range(threads):
        thread = threading.Thread(target=worker, args=(eigenvalue_function, lamb,))
        thread_list.append(thread)

    for thread in thread_list:
        thread.start()

    
    for thread in thread_list:
        thread.join()

    # Collect results from the queue
    sezsez_casov = []
    sezsez_razlik = []
    while not results_queue.empty():
        seznam1, seznam2 = results_queue.get()
        sezsez_casov.append(seznam1)
        sezsez_razlik.append(seznam2)


    #file.close()
    print("done")

    return sezsez_casov, sezsez_razlik


list_of_functions = [diagonalize, jacobi_eigenvalue, qr_algorithm_numpy, LA.eigh, LA.eig]
n = np.arange(1, 21, 10)
# call threads
run_threads(500, n, 0.5, list_of_functions)

#print(sezsez_casov)
#print(sezsez_razlik)


