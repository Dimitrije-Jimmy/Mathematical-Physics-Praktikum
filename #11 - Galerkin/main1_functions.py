import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scs
from scipy.sparse.linalg import spsolve
from scipy.sparse import block_diag
from mpl_toolkits.mplot3d import Axes3D

import time
import threading

import os
dir_path = os.path.dirname(os.path.realpath(__file__))+'\\Images\\Results\\'
print(dir_path)

# Functions ____________________________________________________________________

def testne(m, n, r, phi):
    return r**(2*m + 1) * (1-r)**n * np.sin((2*m + 1) * phi)

def testne_reshaped(m, n, r, phi):
    return (r**(2*m + 1) * (1-r)**n * np.sin((2*m + 1) * phi)).reshape(r.shape)

def Aij(m, N_max):
    i, j = np.indices((N_max, N_max)) + 1
    beta = scs.beta(j + i - 1, 3 + 4*m)
    A = -0.5*beta*np.pi *j*i* (3 + 4*m) / (2 + 4*m + j + i)
    return A
       

def bj(m, N_max):
    j = np.arange(N_max) + 1
    beta = scs.beta(2*m + 3, j + 1)
    b = -2 * beta / (2*m + 1)
    return b


def galerkin(M_max, N_max):
    Aji = []
    b = np.zeros(((M_max + 1) * N_max))
            
    for m in range(M_max+1):
        b[m*N_max : m*N_max + N_max] = bj(m, N_max)
        A = Aij(m, N_max)
        Aji.append(A)
        
    A = block_diag(Aji, format='csr') 
    a = spsolve(A, b)
    C = -(32./np.pi)* np.dot(b, a)
    return C, a


def izracunZ(M_max, N_max):
    Z = np.zeros((N, N))
    for m in range(M_max+1):
        for n in range(1, N_max+1):
            print(f'm = {m}, n = {n}')
            Z += testne(m, n, R, PHI)
    
    return np.array(Z)

"""
def singleZ(m, n):
    #global R, PHI
    return testne(m, n, R, PHI)
"""
# Failed threading
"""
def err_calc(m, n, C100, a100):
    start_time = time.process_time()
    C, a = galerkin(m, n)
    execution_time = start_time - time.process_time()
    err_C = np.abs(C - C100)
    err_a = np.abs(a - a100)
    
    return m, n, C, a, err_C, err_a, execution_time
    

def err_time(M_max_sez, N_max_sez, N):
    # Lock for thread-safe access to the results array
    lock = threading.Lock()

    print("Starting")

    # Create threads
    threads = []
    for m in M_max_sez:
        for n in N_max_sez:
            print(m, n)
            thread = threading.Thread(target=err_calc, args=(m, n, C100, a100, ))
            threads.append(thread)

    # Start the threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()
        
    print("Done")
"""

# Parameters ___________________________________________________________________

N = 1000
M_max = 3
N_max = 3

r = np.linspace(0, 1, N)
phi = np.linspace(0, np.pi, N)
R, PHI = np.meshgrid(r, phi) #za plt.countourf
X = R * np.cos(PHI)
Y = R * np.sin(PHI)


if __name__ == "__main__":
    # Calculations _________________________________________________________________

    
    
    """
    #Plotanje rešitve priporočam m=3, n=3
    C, a = galerkin(M_max, N_max)
    Z = izracunZ(M_max, N_max)
    
    #Za = Z
    # Reshape 'a' to match the shape of 'Z'
    #a_reshaped = a.reshape((4, 3))
    a_reshaped = a.reshape((4, 3, 1, 1))

    # Perform element-wise multiplication
    #Za += np.einsum('ij,kl->ijkl', a_reshaped, Z)
    # Perform element-wise multiplication
    Za = a_reshaped * Z.reshape((1, 1, 1000, 1000))
    """

    
    #Plotanje rešitve priporočam m=3, n=3
    C, a = galerkin(3, 3)
    Z = izracunZ(M_max, N_max)

    
    Za = np.zeros((N, N))
    print(len(a))
    for i in range(M_max+1):
        for j in range(N_max):
            Za += a[i*(3)+j]*testne(i, j+1, R, PHI)
            #print(i, j)
            #print(i*(M_max)+j)
            #Z += a[i*(3)+j]*testne(i, j+1, R, PHI)
            #Za += a[i*(3)+j]*Z[i*(3)+j]
    """      
    a_reshaped = a.reshape((4, 3, 1, 1))
    testne_values = Z.reshape((1, 1, N, N))

    #Za2 = np.sum(a_reshaped * testne_values, axis=(0, 1)) 
    a_reshaped = a.reshape((4, 3, 1, 1))
    Za2 = np.sum(a_reshaped * izracunZ(M_max, N_max), axis=(0, 1))
    
    # Reshape 'a' to match the shape of 'Z' in izracunZ
    a_reshaped = a.reshape((M_max + 1, N_max, 1, 1))
    a_reshaped = a.reshape((M_max + 1, N_max))

    # Use the existing izracunZ function
    #Za2 = izracunZ(M_max, N_max)
    #Za2 *= a_reshaped
    
    #Za2 = np.sum(a_reshaped * izracunZ(M_max, N_max).reshape((M_max + 1, N_max, N, N)), axis=(0, 1))
    Za2 = np.sum(a_reshaped * izracunZ(M_max, N_max), axis=(0, 1))
    
    print(Za.shape)
    print("___")
    print(Za2.shape)
    print(np.sum(np.abs(Za2 - Za)))
    """
    # Plotting _____________________________________________________________________

    plt.contourf(X, Y, Za2, np.linspace(0, 0.105, 500), cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('m_max = 100, n_max = 100')
    plt.show()


    
    import sys
    sys.exit()
    # Grafi pretokov/testnih funkcij _______________________________________
    def testne_plot():
        #fig1, ax1 = plt.figure(1, 1, figsize=(9, 6))
        plt.figure()

        for i in range(M_max+1):
            for j in range(1, N_max+1):
                plt.subplot(M_max+1, N_max, i*(N_max)+j)
                print(i*(N_max)+j)
                Z = np.zeros((N, N))
                for k in range(N):
                    for l in range(N):
                        Z[k][l] = testne(i, j, R[k][l], PHI[k][l])

                plt.contourf(X, Y, Z, cmap='jet')
                plt.colorbar()
                plt.xlabel(r'$x$')
                plt.ylabel(r'$y$')
                plt.title(f'$m$ = {i}, $n$ = {j}')

        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()

    #testne_plot()

    def testne_plot_m(m, N_max, N):

        for n in range(1, N_max+1):
            
            plt.subplot(m, N_max, m+n-1)
            print(f'm = {m}, n = {n}')
            
            Z = np.zeros((N, N))
            for k in range(N):
                for l in range(N):
                    Z[k][l] = testne(m, n, R[k][l], PHI[k][l])

            plt.contourf(X, Y, Z, cmap='viridis')
            plt.colorbar()
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$')
            plt.title(f'$m$ = {m}, $n$ = {n}')
            
            
            """
            plt.subplot(m, N_max, m+n-1)

            plt.contourf(R, PHI, Z, cmap='viridis')
            plt.colorbar()
            plt.xlabel(r'$r$')
            plt.ylabel(r'$\Phi$')
            plt.title(f'$m$ = {m}, $n$ = {n}')
            plt.tight_layout()
            """

        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()


    # plottas za m = 0, 1, 2
    #testne_plot_m(1, N_max, N)





    # Galerkin rešitve _____________________________________________________



    # Error + time ______________________________________________________

    """
    do it v drugi skripti vkljucno z threading
    """


