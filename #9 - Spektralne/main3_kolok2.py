import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm
import numpy.linalg as LA

from scipy.linalg import solve_banded

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 12})

from main1_analytic2 import *

# Functions ____________________________________________________________________

# Kubični B-spline
def Bspline5(xsez, x, k):
    B = np.zeros(len(xsez))

    dx = xsez[1] - xsez[0]

    #for k in range(-1, len(xsez) - 2):
    for i in range(len(xsez)):
        if x[i] <= xsez[k-2]:
            B[i] = 0
        elif xsez[k-2] <= x[i] <= xsez[k-1]:
            B[i] = ((x[i] - xsez[k-2])/dx)**3
        elif xsez[k-1] <= x[i] <= xsez[k]:
            B[i] = ((x[i] - xsez[k-2])/dx)**3 - 4*((x[i] - xsez[k-1])/dx)**3
        elif xsez[k] <= x[i] <= xsez[k+1]:
            B[i] = ((xsez[k-2] - x[i])/dx)**3 - 4*((xsez[k-1] - x[i])/dx)**3
        elif xsez[k+1] <= x[i] <= xsez[k+2]:
            B[i] = ((xsez[k-2] - x[i])/dx)**3
        elif xsez[k+2] <= x[i]:
            B[i] = 0
        else:
            print("why the fuck not work ", k)

    return B


def matrikaA(n):
    A = np.zeros((n, n))
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:, 1:], 1)
    np.fill_diagonal(A, 4)
    A[0, 1] = 1
    A[-1, -2] = 1

    return A

def matrikaB(n):
    global D, a, b, dx
    dx = int(b-a)/n
    
    B = np.zeros((n, n))
    np.fill_diagonal(B[1:], 1)
    np.fill_diagonal(B[:, 1:], 1)
    np.fill_diagonal(B, -2)
    B[0, 1] = 1
    B[-1, -2] = 1

    B = 6*D*B/dx**2
    return B


def Ckoeficienti(n, f):
    global dt
    
    A = matrikaA(n)
    B = matrikaB(n)

    LHS = (A - 0.5*dt*B)
    RHS = (A + 0.5*dt*B)
    
    c = np.zeros((n, n))
    c[0] = np.matmul( LA.inv(A), f )
    
    for i in range(1, len(t)):
        part = np.matmul( LA.inv(LHS), RHS )
        c[i] = np.matmul( part, c[i-1] )
        
    return c
    

def FEM(x, t, n, u0):
    Txt = np.zeros((len(t), len(x)))

    C = Ckoeficienti(len(t), u0)
    for i in range(len(t)):
        print(i)
        for k in range(-1, N-2):
            Txt[i] += -C[i, k]*Bspline5(x, x, k)
            
    return Txt


if __name__ == '__main__':
# Parameters ___________________________________________________________________
    a, b, n, m = 0., 1., 100, 2
    L, D = 1.0, 0.05
    #D = D**2
    sigma = L/4.
    N, M = n, m

    tk = 1
    tn = 100
    tn = n

    x = np.linspace(0, L, n)    # Define x domain
    u0 = np.exp(-((x - L/2.)**2.)/(sigma**2.))

    x2 = np.linspace(0, 2*L*(1. - (1./(2.*N))), N)
    u0 = np.exp(-((x2 - L/2.)**2.)/(sigma**2.)) - np.exp(-((x2 - 3.*L/2.)**2.)/(sigma**2.))  # Dirichletov robni
    
    x = x2
    #u0 = np.sin(np.pi*x)

    t = np.linspace(0, tk, tn)
    dt = t[1] - t[0]


    Txt = FEM(x, t, n, u0)

    # Plotting _____________________________________________________________________
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))


    colors = plt.cm.jet(np.linspace(1, 0, M*N))
    for yind, yval in enumerate(Txt):
        print(yind)
        ax1.plot(x, yval, c=colors[yind])

    ax1.plot(x, Txt[0], 'k--', label='Začetno stanje')
    ax1.legend()
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$T$')
    #ax1.plot(x0, y[0])
    #ax1.plot(x0, zp)
    ax1.set_title(f'Temperatura v odvisnosti od časa - t = {tk} s')
    
    surf2 = ax2.imshow(np.flipud(Txt), cmap='jet', interpolation='nearest')
    #ax2.imshow(y, cmap='jet', interpolation='nearest')
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$t$')
    

    # Add colorbar for reference
    cbar2 = fig.colorbar(surf2, ax=ax2, aspect=10)
    cbar2.set_label(r'Temperatura - $T$ [$^{\circ}$C]')
    
    
    
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()
    """
    
    # Napaka ___________________________________________________________________
    x = np.linspace(0, L, n)    # Define x domain
    u0 = np.exp(-((x - L/2.)**2.)/(sigma**2.))

    x2 = np.linspace(0, 2*L*(1. - (1./(2.*N))), N)
    u0 = np.exp(-((x2 - L/2.)**2.)/(sigma**2.)) - np.exp(-((x2 - 3.*L/2.)**2.)/(sigma**2.))  # Dirichletov robni
    
    x = x2
    u0 = np.sin(np.pi*x)
    t = np.linspace(0, 1., n)
    Txt = FEM(x, t, n, u0)
    
    
    x3 = np.linspace(0, 2*L*(1. - (1./(2.*N))), N)
    print(x3.shape)
    
    #x2 = np.linspace(0, 2*L, n)
    zacetna_fun = 'Gauss'
    zacetna_fun = 'sin'
    yanal = analytic2(x3, t, zacetna_fun=zacetna_fun)
    xind = int(len(yanal[0])/2)
    yanal = yanal[:, :xind]
    Txt = Txt[:, :int(len(Txt[0])/2)]
    y = 2/np.max(Txt)*Txt
    
    print(y.shape)
    print(yanal.shape)
    err = y - yanal
    abserr = abs(err)

    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 6))

    M = int(tk/dt)
    print(N, M)
    colors = plt.cm.jet(np.linspace(1, 0, len(y)))
    
    for yind, yval in enumerate(y):
        print(yind)
        ax1.plot(x[:xind], yval, c=colors[yind])
        
        ax2.plot(x[:xind], err[yind], c=colors[yind])
        
        ax3.plot(x[:xind], yanal[yind], c=colors[yind])
        
        ax4.plot(x[:xind], abserr[yind], c=colors[yind])

    ax1.plot(x[:xind], y[0], 'k--', label='Začetno stanje')
    ax1.legend()
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$T$')
    ax1.set_title(r'Temperatura v odvisnosti od časa')
    
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$Err. = {\rm T} - {\rm T}_{analityc}$')
    ax2.set_title(r'Absolutna napaka v odvisnosti od analitične rešitve')

    ax3.set_xlabel(r'$x$')
    ax3.set_ylabel(r'$T$')
    ax3.set_title(r'Analitična rešitev T(x, t)')
    
    ax2.set_yscale('log')
    ax4.set_xlabel(r'$x$')
    ax4.set_ylabel(r'$Abs. Err. = ||{\rm T} - {\rm T}_{analityc}||$')
    ax4.set_title(r'Absolutna napaka v odvisnosti od analitične rešitve')
    
    
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()
    
        
    import sys
    sys.exit()
    # 3D ___________________________________________________________________________
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plt.set_cmap('jet_r')

    X, T = np.meshgrid(x, t)
    surf = ax.plot_surface(X, T, Txt, cmap='jet', edgecolor='k')    # Dirichletov

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$t$')
    ax.set_zlabel(r'T($x, t$)', rotation=270)
    ax.set_title(r'T($x, t$)')

    # Add colorbar for reference
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.show()