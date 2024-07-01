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


# Eigenvalue function __________________________________________________________
def diagonalize(matrix, tol=10**-10, maxiter=1000):
    tridiag = matrix
    
    s = np.eye(tridiag.shape[0])  # To store eigenvectors

    for i in range(maxiter):
        #print(f"Iter: {i}")
        if i == maxiter - 1:
            #raise Warning("Maximum number of iterations exceeded")
            print("Maximum number of iterations exceeded")
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
def jacobi_eigenvalue(A, tol=1e-10, max_iter=1000):
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


# Time _________________________________________________________________________
import json
import time

#seznam_casov = []
#seznam_razlik = []
def worker(n_sez, lamb):
    global q_matrix_single

    seznam_casov = []
    seznam_razlik = []
    
    for matrix_size in n_sez:
        print("matrix size: ", matrix_size)

        data = anharmonic(lamb, matrix_size, q_matrix_single)

        time_list = []
        eigenvalues_list = []
        
        
        # LA.eig
        start_time = time.time()
        complex_values, _ = LA.eig(data)
        eigenvalues = np.real(complex_values)
        execution_time = time.time() - start_time  
        time_list.append(execution_time)
        # Sort eigenvalues and corresponding eigenvectors in descending order of absolute values
        sort_order = np.argsort(np.abs(eigenvalues))[::-1]  # Reverse the order
        eigenvalues = eigenvalues[sort_order]
        eigenvalues_list.append(eigenvalues)

        # Diagonalize
        start_time = time.time()
        eigenvalues, _ = diagonalize(data)
        #print(eigenvalues)
        execution_time = time.time() - start_time  
        time_list.append(execution_time)
        eigenvalues_list.append(eigenvalues)

        # Jacobi
        start_time = time.time()
        eigenvalues, _ = jacobi_eigenvalue(data)
        #print(eigenvalues)
        execution_time = time.time() - start_time  
        time_list.append(execution_time)
        eigenvalues_list.append(eigenvalues)
        
        # QR numpy
        start_time = time.time()
        eigenvalues, _ = qr_algorithm_numpy(data)
        #print(eigenvalues)
        execution_time = time.time() - start_time  
        time_list.append(execution_time)
        eigenvalues_list.append(eigenvalues)

        # LA.eigh
        start_time = time.time()
        eigenvalues, _ = LA.eigh(data)
        #print(eigenvalues)
        execution_time = time.time() - start_time  
        time_list.append(execution_time)
        # Sort eigenvalues and corresponding eigenvectors in descending order of absolute values
        sort_order = np.argsort(np.abs(eigenvalues))[::-1]  # Reverse the order
        eigenvalues = eigenvalues[sort_order]
        eigenvalues_list.append(eigenvalues)

        
        #print(time_list)
        #print(eigenvalues_list)
        data_array = np.array(eigenvalues_list)
        # Calculate the element-wise absolute differences with the last sublist
        differences = np.abs(data_array - data_array[0])
        # Sum the absolute differences along the rows
        razlika_lastnih_vrednosti = np.sum(differences, axis=1)
        #print(razlika_lastnih_vrednosti)
        
        #file.write(f"{matrix_size},\t {x_sez},\t {y_sez},\t {time1},\t {time2},\t {theta},\t {l},\t {razdalja1}\n")
        #file.flush()

        #matrix.append([x_sez, y_sez, time1, time2, theta, l, razdalja1])

        seznam_casov.append(time_list)
        seznam_razlik.append(razlika_lastnih_vrednosti)

    #print(seznam_casov)
    #sezsez_casov.append(seznam_casov)
    #sezsez_razlik.append(seznam_razlik)

    return np.array(seznam_casov), np.array(seznam_razlik)

# To not override my files
import datetime
current_time = str(datetime.datetime.now())
print(current_time)
current_time_fixed = current_time.split('.')[0]
current_time_fixed = current_time_fixed.replace('-', '_').replace(':', '.')


directory = os.path.dirname(__file__)+'\\Meritve\\'
file_path = directory + f'meritev_{current_time_fixed}'
print(file_path)


list_of_functions = [LA.eig, diagonalize, jacobi_eigenvalue, qr_algorithm_numpy, LA.eigh]
n_sez = [1] + np.arange(10, 110, 10)
lamb = 0.5
# call threads
sez_casov, sez_razlik = worker(n_sez, lamb)

data = {
    "lambda": lamb,
    "execution time": sez_casov.tolist(),
    "difference": sez_razlik.tolist()
}
with open(file_path+'.json', 'w') as json_file:
    json.dump(data, json_file)


print(sez_casov)
print(sez_razlik)


# Plotting _____________________________________________________________________
functions = ['eig - Numpy', 'Householder', 'Jacobi', 'QR - Numpy', 'eigh - Numpy']
cmap = plt.get_cmap('viridis')
barve = [cmap(value) for value in np.linspace(0, 1, 5)]


plt.figure(figsize=(12, 6))

"""
for index, t in np.ndenumerate(sez_casov):
    i, j = index
    plt.scatter(n_sez[i], t, label=f'{functions[j]}', color=barve[j])
    plt.plot(n_sez[i], t, alpha=0.5, lw=0.3, color=barve[j])
"""
plt.scatter(n_sez, sez_casov[:, 0], label=f'{functions[0]}', color=barve[0])
plt.plot(n_sez, sez_casov[:, 0], alpha=0.5, lw=0.3, color=barve[0])
plt.scatter(n_sez, sez_casov[:, 1], label=f'{functions[1]}', color=barve[1])
plt.plot(n_sez, sez_casov[:, 1], alpha=0.5, lw=0.3, color=barve[1])
plt.scatter(n_sez, sez_casov[:, 2], label=f'{functions[2]}', color=barve[2])
plt.plot(n_sez, sez_casov[:, 2], alpha=0.5, lw=0.3, color=barve[2])
plt.scatter(n_sez, sez_casov[:, 3], label=f'{functions[3]}', color=barve[3])
plt.plot(n_sez, sez_casov[:, 3], alpha=0.5, lw=0.3, color=barve[3])
plt.scatter(n_sez, sez_casov[:, 4], label=f'{functions[4]}', color=barve[4])
plt.plot(n_sez, sez_casov[:, 4], alpha=0.5, lw=0.3, color=barve[4])


plt.title(f'Časovna zahtevnost posameznih metod, $\lambda = {lamb}$')
plt.xlabel(r'Velikost matrike $N\times N$')
plt.ylabel(r'Čas izračuna lastnih vrednosti $t\;[s]$')
plt.legend()

plt.savefig(file_path+'_time.png', dpi=300)
plt.show()


plt.figure(figsize=(12, 6))

"""
for i in range(1, len(n_sez[0])):
    plt.scatter(n_sez, sez_casov[:, i], label=f'{functions[i]}', color=barve[i])
    plt.plot(n_sez, sez_casov[:, i], alpha=0.5, lw=0.3, color=barve[i])
"""
#plt.scatter(n_sez, sez_casov[:, 0], label=f'{functions[0]}', color=barve[0])
#plt.plot(n_sez, sez_casov[:, 0], alpha=0.5, lw=0.3, color=barve[0])
plt.scatter(n_sez, sez_casov[:, 1], label=f'{functions[1]}', color=barve[1])
plt.plot(n_sez, sez_casov[:, 1], alpha=0.5, lw=0.3, color=barve[1])
plt.scatter(n_sez, sez_casov[:, 2], label=f'{functions[2]}', color=barve[2])
plt.plot(n_sez, sez_casov[:, 2], alpha=0.5, lw=0.3, color=barve[2])
plt.scatter(n_sez, sez_casov[:, 3], label=f'{functions[3]}', color=barve[3])
plt.plot(n_sez, sez_casov[:, 3], alpha=0.5, lw=0.3, color=barve[3])
plt.scatter(n_sez, sez_casov[:, 4], label=f'{functions[4]}', color=barve[4])
plt.plot(n_sez, sez_casov[:, 4], alpha=0.5, lw=0.3, color=barve[4])

plt.title(f'Absolutna napaka v odvisnosti od $numpy.linalg.eig$, $\lambda = {lamb}$')
plt.xlabel(r'Velikost matrike $N\times N$')
plt.ylabel(r'$E - E_{ref}$')
plt.legend()

plt.savefig(file_path+'_diff.png', dpi=300)
plt.show()
