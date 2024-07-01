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

def q_matrix_double(n):
    matrix = np.zeros((n, n))
    for ind, val in np.ndenumerate(matrix):
        i, j = ind
        a = np.sqrt(j * (j - 1)) * delta(i, j - 2)
        b = (2 * j + 1) * delta(i, j)
        c = np.sqrt((j + 1) * (j + 2)) * delta(i, j + 2)
        matrix[i][j] = 0.5 * (a + b + c)

    return np.matmul(matrix, matrix)

def q_matrix_quad(n):
    matrix = np.zeros((n, n))
    for ind, val in np.ndenumerate(matrix):
        i, j = ind
        prefac = 1/(2**4) * np.sqrt((2**i * np.math.factorial(i))/(2**j * np.math.factorial(j)))
        a = delta(i, j + 4)
        b = 4*(2 * j + 3) * delta(i, j + 2)
        c = 12*(2*j**2 + 2*j + 1) * delta(i, j)
        d = 16*j*(2*j**2 - 3*j + 1) * delta(i, j - 2)
        e = 16*j*(j**3 - 6*j**2 + 11*j - 6) * delta(i, j - 4)
        matrix[i][j] = prefac * (a + b + c + d + e)

    return matrix


def basis(q, n):
    return (2**n * np.math.factorial(n) * np.sqrt(np.pi))**(-1/2) * np.exp(-q**2/2) * scipy.special.eval_hermite(n, q)

def anharmonic(lamb, n, func):
    return lho(n) + lamb*func(n)


def plot_poly(x, vect):
    c = [vect[i]*basis(x, i) for i in range(len(vect))]
    return np.sum(c)


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

        data1 = anharmonic(lamb, matrix_size, q_matrix_single)
        data2 = anharmonic(lamb, matrix_size, q_matrix_double)
        data3 = anharmonic(lamb, matrix_size, q_matrix_quad)

        time_list = []
        eigenvalues_list = []
        
        
        # Single _______________________________________________________
        start_time = time.time()
        complex_values, _ = LA.eig(data1)
        eigenvalues = np.real(complex_values)
        execution_time = time.time() - start_time  
        time_list.append(execution_time)
        # Sort eigenvalues and corresponding eigenvectors in descending order of absolute values
        sort_order = np.argsort(np.abs(eigenvalues))[::-1]  # Reverse the order
        eigenvalues = eigenvalues[sort_order]
        eigenvalues_list.append(eigenvalues)

        # Double _______________________________________________________
        start_time = time.time()
        complex_values, _ = LA.eig(data2)
        eigenvalues = np.real(complex_values)
        execution_time = time.time() - start_time  
        time_list.append(execution_time)
        # Sort eigenvalues and corresponding eigenvectors in descending order of absolute values
        sort_order = np.argsort(np.abs(eigenvalues))[::-1]  # Reverse the order
        eigenvalues = eigenvalues[sort_order]
        eigenvalues_list.append(eigenvalues)
        
        # Quadruple ____________________________________________________
        start_time = time.time()
        complex_values, _ = LA.eig(data3)
        eigenvalues = np.real(complex_values)
        execution_time = time.time() - start_time  
        time_list.append(execution_time)
        # Sort eigenvalues and corresponding eigenvectors in descending order of absolute values
        sort_order = np.argsort(np.abs(eigenvalues))[::-1]  # Reverse the order
        eigenvalues = eigenvalues[sort_order]
        eigenvalues_list.append(eigenvalues)

        
        #print(time_list)
        #print(eigenvalues_list)
        data_array = np.array(eigenvalues_list)
        differences = np.abs(data_array - data_array[0])
        razlika_lastnih_vrednosti = np.sum(differences, axis=1)
        #print(razlika_lastnih_vrednosti)

        seznam_casov.append(time_list)
        seznam_razlik.append(razlika_lastnih_vrednosti)


    return np.array(seznam_casov), np.array(seznam_razlik)


# To not override my files (ghastly mistake) ___________________________
import datetime
current_time = str(datetime.datetime.now())
print(current_time)
current_time_fixed = current_time.split('.')[0]
current_time_fixed = current_time_fixed.replace('-', '_').replace(':', '.')


directory = os.path.dirname(__file__)+'\\Meritve\\'
file_path = directory + f'meritev_{current_time_fixed}'
print(file_path)


n_sez = np.arange(5, 150, 5)
lamb = 0
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
functions = ['[q]^4', '[q^2]^2', '[q^4]']
cmap = plt.get_cmap('viridis')
barve = [cmap(value) for value in np.linspace(0, 1, 3)]


plt.figure(figsize=(12, 6))

plt.scatter(n_sez, sez_casov[:, 0], label=f'${functions[0]}$', color=barve[0])
plt.plot(n_sez, sez_casov[:, 0], alpha=0.5, lw=0.3, color=barve[0])
plt.scatter(n_sez, sez_casov[:, 1], label=f'${functions[1]}$', color=barve[1])
plt.plot(n_sez, sez_casov[:, 1], alpha=0.5, lw=0.3, color=barve[1])
plt.scatter(n_sez, sez_casov[:, 2], label=f'${functions[2]}$', color=barve[2])
plt.plot(n_sez, sez_casov[:, 2], alpha=0.5, lw=0.3, color=barve[2])

#plt.xscale('log')
#plt.yscale('log')

plt.title(f'Časovna zahtevnost posameznih metod, $\lambda = {lamb}$')
plt.xlabel(r'Velikost matrike $N\times N$')
plt.ylabel(r'Čas izračuna lastnih vrednosti $t\;[s]$')
plt.legend()

plt.savefig(file_path+'_time.png', dpi=300)
plt.show()


plt.figure(figsize=(12, 6))

#plt.scatter(n_sez, sez_casov[:, 0], label=f'${functions[0]}$', color=barve[0])
#plt.plot(n_sez, sez_casov[:, 0], alpha=0.5, lw=0.3, color=barve[0])
plt.scatter(n_sez, sez_casov[:, 1], label=f'${functions[1]}$', color=barve[1])
plt.plot(n_sez, sez_casov[:, 1], alpha=0.5, lw=0.3, color=barve[1])
plt.scatter(n_sez, sez_casov[:, 2], label=f'${functions[2]}$', color=barve[2])
plt.plot(n_sez, sez_casov[:, 2], alpha=0.5, lw=0.3, color=barve[2])

#plt.xscale('log')
#plt.yscale('log')

plt.title(f'Absolutna napaka v odvisnosti od $[q]^4$, $\lambda = {lamb}$')
plt.xlabel(r'Velikost matrike $N\times N$')
plt.ylabel(r'$E - E_{ref}$')
plt.legend()

plt.savefig(file_path+'_diff.png', dpi=300)
plt.show()
