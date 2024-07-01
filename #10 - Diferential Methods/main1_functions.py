import numpy as np
import matplotlib.pyplot as plt
#from cmath import *
import cmath as cm
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded
from scipy.sparse import csr_matrix

import matplotlib.cm as cm
import matplotlib.colors as mcolors


# Functions ____________________________________________________________________

def Vpot(x):
    global k
    return 0.5*k*x**2


def analytic_koherent(x, t):
    """
    #Use this one cause it's correct
    IN:
    x - numpy.ndarray
    t - numpy.ndarray
    k - koeficient potenciala
    
    OUT:
    Psi - numpy.ndarray matrix
    """
    #global w, lamb, a, b
    
    #w = k**0.5
    w = 0.2
    k = w**2
    lamb = 10.
    alpha = k**0.25
    
    a = x[0]
    b = x[-1]
    
    M = len(t)
    N = len(x)
    
    xi = alpha*x
    xiL = alpha*lamb
    """
    Psi = np.array((M, N))
    for i, tval in enumerate(t):
        factor1 = cm.sqrt(alpha*cm.sqrt(cm.pi))
        factor2 = -0.5*(xi - xiL*cm.cos(w*tval))**2
        factor3 = -1j(w*t/2 + xi*xiL*cm.sin(w*tval) - 0.25*xiL**2*cm.sin(2*w*tval))
        #z = complex(factor2, factor3)
        #Psi[p] = factor1*np.exp(z)
        Psi[i] = factor1*np.exp(factor2 + factor3)
    """
    Psi = np.array((M, N), dtype=np.complex128)
    for i, tval in enumerate(t):
        factor1 = np.sqrt(alpha*np.sqrt(np.pi))
        factor2 = -0.5*(xi - xiL*np.cos(w*tval))**2
        factor3 = -1j*(w*tval/2 + xi*xiL*np.sin(w*tval) - 0.25*xiL**2*np.sin(2*w*tval))
        f = factor1*np.exp(factor2 + factor3)
        print(f)
        Psi[i] = f
    
    return Psi

def analytic_koherent2(x, t, k=1.):
    """
    IN:
    x - numpy.ndarray
    t - scalar, value of t at which to evaluate
    k - koeficient potenciala
    
    OUT:
    Psi - numpy.ndarray matrix
    """
    global w, lamb, a, b
    
    xi = alpha*x
    xiL = alpha*lamb
    
    t_matrix, xi_matrix = np.meshgrid(t, xi)
    
    factor1 = np.sqrt(alpha*np.sqrt(np.pi))
    factor2 = -0.5*(xi_matrix - xiL*np.cos(w*t_matrix))**2
    factor3 = -1j*(w*t_matrix/2 + xi_matrix*xiL*np.sin(w*t_matrix) - 0.25*xiL**2*np.sin(2*w*t_matrix))

    Psi = factor1*np.exp(factor2 + factor3)
    
    return Psi.T


def analytic_empty(x, t):
    global sigma0, k0, lamb2
    
    lamb = lamb2
    
    Psi = np.array((M, N))
    for i, tval in enumerate(t):
        factor0 = 1 + 1j*tval/(2*sigma0**2)
        factor1 = ((2*np.pi*sigma0**2)**(-1/4)) / np.sqrt(factor0)
        factor2 = -((x-lamb) / (2*sigma0))**2 + 1j*k0*(x-lamb) - 0.5j*tval*k0**2
        Psi[i] = factor1*np.exp(factor2/factor0)
    
    return Psi


def analytic_empty2(x, t):
    global sigma0, k0, lamb2
    
    lamb = lamb2
    
    t_matrix, x_matrix = np.meshgrid(t, x)
    
    factor0 = 1 + 1j*t_matrix/(2*sigma0**2)
    factor1 = ((2*np.pi*sigma0**2)**(-1/4)) / np.sqrt(factor0)
    factor2 = -((x_matrix-lamb) / (2*sigma0))**2 + 1j*k0*(x_matrix-lamb) - 0.5*1j*t_matrix*k0**2
    Psi = factor1*np.exp(factor2/factor0)
    
    return Psi.T


def valovna_prvi(x):
    global alpha, lamb
    Psi0 = np.sqrt(alpha*np.sqrt(np.pi)) * np.exp( -0.5*(alpha*(x - lamb))**2 )  
    return Psi0

def valovna_drugi(x):
    global sigma0, lamb2, k0
    Psi0 = (2*np.pi*sigma0**2)**(-0.25) * np.exp( 1j*k0*(x - lamb2) ) * np.exp( -((x - lamb2)/(2*sigma0))**2 )  
    return Psi0


"""
def FTCS(x, t, vf, K=1.):

    #LMAO to je FTCS za toplotno / difuzijsko enačbo
 
    #global k
    
    a = x[0]
    b = x[-1]
    t0 = t[0]
    tf = t[-1]
    
    M = len(t)
    N = len(x)
    
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    r = K * dt / (dx * dx)
    
    Psi = np.zeros((M, N), dtype=np.complex_)
    
    # Initial conditions
    Psi[0, :] = vf
    # Boundary conditions
    Psi[:, 0] = 0
    Psi[:, -1] = 0
    
    for i in range(M-1):
        Psi[i+1, 1:-1] = r * Psi[i, 0:-2] + ( 1.0 - 2 * r ) * Psi[i, 1:-1] + r * Psi[i, 2:]
    
    return Psi

def FTCS2(x, t, vf, K=1.):
    a = x[0]
    b = x[-1]
    t0 = t[0]
    tf = t[-1]
    
    M = len(t)
    N = len(x)
    
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    r = K * dt / (dx * dx)
    
    Psi = np.zeros((M, N), dtype=np.complex_)
    
    # Initial conditions
    Psi[0, :] = vf
    # Boundary conditions
    Psi[:, 0] = 0
    Psi[:, -1] = 0
    
    b = 0.5j*dt/(dx * dx)
    a = -0.5*b
    V = Vpot(x)
    d = 1.0 + b + 0.5j*dt*V
    
    for i in range(M-1):
        #Psi[i+1, 1:-1] = r * Psi[i, 0:-2] + ( 1.0 - 2 * r ) * Psi[i, 1:-1] + r * Psi[i, 2:]
        Psi[i+1, 1:-1] = a*r * Psi[i, 0:-2] + d[1:-1]*r * Psi[i, 1:-1] + a*r * Psi[i, 2:]
    
    return Psi
"""


def matrix_A(x, t):
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    b = 0.5j*dt/(dx * dx)
    a = -0.5*b
    V = Vpot(x)
    d = 1.0 + b + 0.5j*dt*V

    #dim = d.shape[0]
    dim = len(x)
    #print(dim)
    diagonals = [d, a, a]
    matrikaA = sparse.diags(diagonals, [0, -1, 1], shape=(dim, dim), format='csr')#.toarray()
    
    #matrikaA = sparse.diags(diagonals, [0, -1, 1], shape=(dim, dim)).toarray()
    #matrikaA = np.diag(d, 0) + np.diag(-0.5*np.ones(len(x)-1)*b, 1) + np.diag(-0.5*np.ones(len(x)-1)*b, -1)

    #matrikaA = csr_matrix(matrikaA)

    return matrikaA
    
    
def C_N(x, t, vf):
    """
    Crank-Nicolson method
    IN:
    x - ndarray
    t - ndarray
    vf - ndarray - starting 
    
    OUT:
    Psi - ndmatrix
    """
    M = len(t)
    N = len(x)

    Psi = np.zeros((M, N), dtype=np.complex_)
    Psi[0] = vf
    A = matrix_A(x, t)
    for i in range(M):
        rhs = np.conjugate(A).dot(vf)
        vf = spsolve(A, rhs)
        #vf = solve_banded((1, 1), A, rhs)
        Psi[i] = vf
    
    return Psi


# Initialization _______________________________________________________________

# prvi del _____________________________________________________________
#k = 1.
#w = k**0.5

w = 0.2
k = w**2.
lamb = 10.
alpha = k**(1/4)

"""
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
"""


# drugi del ____________________________________________________________
sigma0 = 1./20.
k0 = 50.*np.pi
lamb2 = 0.25

"""
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
"""


# Plotting _____________________________________________________________________

if __name__ == '__main__':
    print(None)
    
    N = 300
    M = 1000

    # prvi del
    a = -40.
    b = 40.
    t0 = 0.
    tnih = 2*np.pi / w
    tf = 314.
    tf = 0.5*tnih
    
    dx = (b - a)/(N-1)
    dt = (tf - t0)/(M-1)

    x = np.linspace(a, b, N)
    t = np.linspace(t0, tf, M)
    #t = np.arange(t0, tf, dt)
    
    
    # drugi del
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
    
    
    phi1_anal = analytic_koherent2(x, t)
    
    # The second one is not working correctly
    phi2_anal = analytic_empty2(x2, t2)
    
    
    zp3 = valovna_prvi(x)
    phi3 = C_N(x, t, zp3)
    
    #zp4 = valovna_drugi(x2)
    #phi4 = C_N(x2, t2, zp4)
    
    phi = phi1_anal
    phi_anal = phi1_anal
    #phi = phi4
    #phi_anal = phi2_anal
    #x = x2
    #t = t2
    

           
    plt.figure(figsize=(8, 6))                                          # PLOT 1
    plt.title('Začetni pogoj: Gauss')
    plt.plot(x, np.real(phi[0]), 'b', label='Numerical')
    plt.plot(x, np.real(zp3), 'r', label='Analytical')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$Re(\Psi)$')
    plt.legend()
    plt.grid(True)
    #plt.show()


    # ChatGPT popravki
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

    q1 = ax1.imshow(np.real(phi), cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x[0], x[-1], t[0], t[-1]])
    ax1.set_title('Real part')
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$t$')

    q2 = ax2.imshow(np.imag(phi), cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x[0], x[-1], t[0], t[-1]])
    ax2.set_title('Imaginary part')
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$t$')

    phisq = np.abs(phi**2)
    q3 = ax3.imshow(phisq, cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x[0], x[-1], t[0], t[-1]])
    ax3.set_title(r'$|\Psi|^2$')
    ax3.set_xlabel(r'$x$')
    ax3.set_ylabel(r'$t$')

    plt.colorbar(q1, ax=ax1, label='Real')
    plt.colorbar(q2, ax=ax2, label='Imaginary')
    cbar3 = plt.colorbar(q3, ax=ax3, label=r'$|\Psi|^2$')
    cbar3.set_label('Time', rotation=270, labelpad=15)

    plt.suptitle('Analitična rešitev: Gauss')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

    colors = plt.cm.jet(np.linspace(1, 0, M))
    for i in range(M):
        ax1.plot(x, np.real(phi[i]), color=colors[i])
    ax1.set_title('Real part')
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$Re(\Psi)$')

    for i in range(M):
        ax2.plot(x, np.imag(phi[i]), color=colors[i])
    ax2.set_title('Imaginary part')
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$Im(\Psi)$')

    for i in range(M):
        phisq = np.abs(phi[i]**2)
        ax3.plot(x, phisq, color=colors[i])
    ax3.set_title(r'$|\Psi|^2$')
    ax3.set_xlabel(r'$x$')
    ax3.set_ylabel(r'$|\Psi|^2$')

    # Create a ScalarMappable
    sm = cm.ScalarMappable(cmap=cm.jet_r, norm=mcolors.Normalize(vmin=t[0], vmax=t[-1]))
    sm.set_array([])  # An array with no data to normalize

    cbar = plt.colorbar(sm, ax=ax3, label='Time')
    cbar.set_label('Time', rotation=270, labelpad=15)

    plt.tight_layout()
    
    
    # napaka ___________________________________________________________

    napaka1 = np.abs(phi - phi_anal)**2

    phisq = phi**2
    phi_analsq = phi_anal**2
    napaka2 = np.abs(phisq - phi_analsq)
    
    napaka = napaka1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    q = ax1.imshow(napaka, cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x[0], x[-1], t[0], t[-1]])
    ax1.set_title(r'Absolutna napaka $|\Psi|^2$')
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$t$')
    
    #plt.colorbar(q, ax=ax1, aspect=10, label=r'$||\Psi|^2 - |\Psi_{analytic}|^2|$')
    plt.colorbar(q, ax=ax1, label=r'$||\Psi|^2 - |\Psi_{analytic}|^2|$')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    

    for i in range(M):
        ax2.plot(x, napaka[i], color=colors[i])
    ax2.set_title(r'$|\Psi|^2$')
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$|\Psi|^2$')

    # Create a ScalarMappable
    sm = cm.ScalarMappable(cmap=cm.jet_r, norm=mcolors.Normalize(vmin=t[0], vmax=t[-1]))
    sm.set_array([])  # An array with no data to normalize

    cbar = plt.colorbar(sm, ax=ax2, label='Time')
    cbar.set_label('Time', rotation=270, labelpad=15)

    plt.tight_layout()

    plt.title('Analitična rešitev: Gauss')

    plt.show()