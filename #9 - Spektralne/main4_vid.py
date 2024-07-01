import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 12})

from main1_analytic2 import *

"""
\frac{\partial T_k(t)}{\partial t} = D(-2\pi fk)^2 T_k(t)

integracija:
T_k(t + h) = T_k(t) + h*D(-2\pi fk)^2 T_k(t)

analitično:
T_k(t) = C*exp(-4pi^2 fk^2 D t)

T(x, t) = ifft (T_k(t))

fk = k/a, 0<x<a

"""

"""
u_t = \alpha^2 u_xx
| Fourier
\hat{u}_t = -\alpha^2 \kappa^2 \hat{u}  <-- RHS
"""
def rhsHeat(uhat_ri, t, kappa, a):
    uhat = uhat_ri[:N] + (1j) * uhat_ri[N:]
    d_uhat = -a**2 * (np.power(kappa, 2)) * uhat
    d_uhat_ri = np.concatenate((d_uhat.real, d_uhat.imag)).astype('float64')
    return d_uhat_ri


def FastFourier(x, t, n, u0):
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    u0hat = np.fft.fft(u0)
    
    # Define discrete wavenumbers
    kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)
    
    # SciPy's odeint function doesn't play well with complex numbers so we are
    # the state u0hat from an N-element complex vector to  a 2N-element real vector
    u0hat_ri = np.concatenate((u0hat.real, u0hat.imag))
    
    uhat_ri = odeint(rhsHeat, u0hat_ri, t, args=(kappa, a))
    
    uhat = uhat_ri[:, :N] + (1j) * uhat_ri[:, N:]
    
    u = np.zeros_like(uhat)
    
    for k in range(len(t)):
        u[k, :] = np.fft.ifft(uhat[k, :])
    
    u = u.real
    
    return u


if __name__ == '__main__':
# Parameters ___________________________________________________________________
    D = 0.05       # Thermal diffusivity constant
    a = np.sqrt(D)
    L = 1.     # Length of domain
    N = 100    # Number of discretization points
    dx = L/N
    x = np.arange(-L/2, L/2, dx)    # Define x domain
    x = np.arange(0, L, dx)    # Define x domain


    # Initial conditions for rectangle box
    #u0 = np.zeros_like(x)
    #u0[int((L/2 - L/10)/dx):int((L/2 + L/10)/dx)] = 1
    #u0hat = np.fft.fft(u0)


    sigma = L/4.
    u0 = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-((x - L/2.)/(2*sigma))**2.)
    u0 = np.exp(-((x - L/2.)/(sigma))**2.)
    #u0 = np.exp(-((x)/sigma)**2.)

    x2 = np.linspace(0, 2*L*(1. - (1./(2.*N))), N)
    u0 = np.exp(-((x2 - L/2.)**2.)/(sigma**2.)) - np.exp(-((x2 - 3.*L/2.)**2.)/(sigma**2.))  # Dirichletov robni
    
    x=x2
    #u0 = np.sin(np.pi*x)
    
    zacetna_fun = 'Gauss'
    #zacetna_fun = 'sin'

    # Simulate in Fourier frequency domain
    dt = 0.005
    tk = 1
    t = np.arange(0, tk, dt)

    u = FastFourier(x, t, N, u0)
    y = 2*u
    
    yanal = analytic(x, t, zacetna_fun=zacetna_fun)
    err = y - yanal
    abserr = abs(err)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 6))

    M = int(tk/dt)
    print(N, M)
    colors = plt.cm.jet(np.linspace(1, 0, M))
    
    for yind, yval in enumerate(y):
        print(yind)
        ax1.plot(x, yval, c=colors[yind])
        
        ax2.plot(x, err[yind], c=colors[yind])
        
        #ax3.plot(x, yanal[yind], c=colors[yind])
        
        #ax4.plot(x, abserr[yind], c=colors[yind])

    ax1.plot(x, y[0], 'k--', label='Začetno stanje')
    ax1.legend()
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$T$')
    ax1.set_title(r'Temperatura v odvisnosti od časa')
    
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$Err. = {\rm T} - {\rm T}_{analityc}$')
    ax2.set_title(r'Absolutna napaka v odvisnosti od analitične rešitve')

    
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()


    import sys
    sys.exit()
    # Waterfall plot ___________________________________________________________
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.set_cmap('jet_r')
    u_plot = u[0:-1:10, :]
    tk = 20
    u_plot = u[::tk, :]     # this is just making the discrete amount of samples
    """
    #for j in range(u_plot.shape[0]):
    for j in range(0, 10):
        ys = j*np.ones(u_plot.shape[1])
        #ax.plot(x, ys, u_plot[j, :], color=cm.jet_r(j*tk*2))
        ind = int(len(x)/2)
        ax.plot(x[:ind], ys[:ind], u_plot[j, :ind], color=cm.jet_r(j*tk*2))    # Dirichletov
    """
    # For continous plot this is better
    u_plot = u             # Continous plot, adjust colors
    
    X, T = np.meshgrid(x, t)
    ind = int(len(u_plot)/2)
    surf = ax.plot_surface(X, T, u_plot, cmap='jet')    # Dirichletov
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Temperature')
    #"""
    #ax.set_ylim((0, 6))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$t$')
    ax.set_zlabel(r'T($x, t$)', rotation=270)
    ax.set_title(r'Temperatura - T($x$, $t$)')
    
    """
    # Image plot
    plt.figure()
    #plt.imshow(np.flipud(u), aspect=8)
    plt.imshow(np.flipud(u[:, :int(len(u)/2)]), aspect=8)      # Dirichletov
    #plt.imshow(u, aspect=8)

    plt.set_cmap('jet')

    #plt.axis('off')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title(r'T($x, t$)')
    """
    plt.show()