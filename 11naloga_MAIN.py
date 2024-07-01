import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import timeit as timeit
from scipy.integrate import solve_bvp
import copy
from scipy import linalg as la
#from bvp import *
import numpy
from scipy import integrate
import scipy.special as scs
from scipy.sparse.linalg import spsolve
from scipy.sparse import block_diag

import os
dir_path = os.path.dirname(os.path.realpath(__file__))+'\\Images\\Manca + Spenko'
print(dir_path)

#--------------------------------------------------------------------
#Sestava funkcij za pomoč izračuna sistema enačb

def trial_func(m, n, r, phi):
    return r**(2*m + 1) * (1-r)**n * numpy.sin((2*m + 1) * phi)

#sestava podvektorja
def bj(m, nmax):
    bj_temp = numpy.zeros(nmax)
    for i in range(nmax):
        n = i+1
        bj_temp[i] = -2.*scs.beta(2*m+3., n+1.)/(2.*m +1.)
    return bj_temp

#matriko razdelimo na bloke zaradi boljše preglednosti indeksov
#sestava bločne matrike(simetrična)
#samo diagonalni bloki delta(m'm)
def Aij(m, nmax):
    Aij_temp = numpy.zeros((nmax, nmax))
    for i in range(nmax):
        for j in range(i+1):
            Aij_temp[i][j] = -(numpy.pi/2.)*(i+1.)*(j+1.)*(3.+4.*m)*scs.beta(i+j+1.,3.+4.*m)/(2.+4.*m+i+j+2.)
            if(j != i): Aij_temp[j][i] = Aij_temp[i][j]
    return Aij_temp

#--------------------------------------------------------------------
#Glavni program (Metoda Galerkina):
#Izracun referencne vrednosti:

N = 1000

r = numpy.linspace(0, 1, N)
phi = numpy.linspace(0, numpy.pi, N)
R, PHI = numpy.meshgrid(r, phi) #za plt.countourf
X = R * numpy.cos(PHI)
Y = R * numpy.sin(PHI)
Z = numpy.zeros((N, N))

def galerkin(mmax, nmax):
    a = numpy.zeros((mmax+1)*nmax)
    b = numpy.zeros((mmax+1)*nmax)
    A_temp = []
    for i in range(mmax+1):
        b[i*(nmax):(i+1)*(nmax)] = bj(i, nmax)
        A_temp.append(Aij(i, nmax))
    A = block_diag(A_temp, format='csr')
    a = spsolve(A, b)
    C = -(32./numpy.pi)*b.dot(a)
    return C, a

#Plotanje rešitve priporočam m=3, n=3
C, a = galerkin(100, 100)
for i in range(101):
    for j in range(100):
        print(i, j)
        for k in range(N):
            for l in range(N):
                Z[k][l] += a[i*(3)+j]*trial_func(i, j+1, R[k][l], PHI[k][l])


plt.contourf(X, Y, Z, numpy.linspace(0, 0.105, 500), cmap='jet')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('m_max = 100, n_max = 100')
plt.show()

#izracun odvisnosti error od dimenzij (m,n)
mmax = [0,1,2,5,10,20,30,50,75,100]
nmax = 50
C_ref, a_ref = galerkin(100, 100)

plt.figure()
for i in mmax:
    error = []
    for j in range(1, nmax):
        C = galerkin(i, j+1)[0]
        print(C)
        error.append(numpy.abs(C - C_ref))
    #plt.plot(range(1, nmax), error, label='m_{max} = %d' % i)
    plt.semilogy(range(1, nmax), error, label='m_{max} = %d' % i)
plt.legend(loc=(0.75,0.7))
plt.xlabel('$n_{max}$')
plt.ylabel('$|C_{100, 100} - C_{m_{max},n_{max}}|$')
plt.show()

#--------------------------------------------------------------------
#Prikaz testnih funkcij:

N = 1000
mmax = 3
nmax = 3

r = numpy.linspace(0, 1, N)
phi = numpy.linspace(0, numpy.pi, N)
R, PHI = numpy.meshgrid(r, phi) #za plt.countourf
X = R * numpy.cos(PHI)
Y = R * numpy.sin(PHI)

plt.figure()
for i in range(mmax+1):
    for j in range(1, nmax+1):
        plt.subplot(mmax+1, nmax, i*(nmax)+j)
        print(i*(nmax)+j)
        Z = numpy.zeros((N, N))
        for k in range(N):
            for l in range(N):
                Z[k][l] = trial_func(i, j, R[k][l], PHI[k][l])
        plt.contourf(X, Y, Z, cmap='jet')
        plt.colorbar()
        #plt.xlabel('x')
        #plt.ylabel('y')
        plt.title('m = ' + str(i) + ', n = ' + str(j))
plt.show()

#------------------------------------------------------------------------------------------
