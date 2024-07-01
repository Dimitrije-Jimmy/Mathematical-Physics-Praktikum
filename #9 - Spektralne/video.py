import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 12})

"""
\frac{\partial T_k(t)}{\partial t} = D(-2\pi fk)^2 T_k(t)

integracija:
T_k(t + h) = T_k(t) + h*D(-2\pi fk)^2 T_k(t)

analitično:
T_k(t) = C*exp(-4pi^2 fk^2 D t)

T(x, t) = ifft (T_k(t))

fk = k/a, 0<x<a

"""

a = 0.05       # Thermal diffusivity constant
L = 1     # Length of domain
N = 100    # Number of discretization points
dx = L/N
x = np.arange(-L/2, L/2, dx)    # Define x domain
x = np.arange(0, L, dx)    # Define x domain

# Define discrete wavenumbers
kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)

# Initial conditions for rectangle box
#u0 = np.zeros_like(x)
#u0[int((L/2 - L/10)/dx):int((L/2 + L/10)/dx)] = 1
#u0hat = np.fft.fft(u0)
"""
a, b, n, m = 0., 1., 2, 100
L, D = 1.0, 0.05
sigma = L/4.
x = np.linspace(a, b, m)
t0 = np.linspace(0., 1., m*n)

u0 = np.exp(-((x - L/2.)**2.)/(sigma**2.))
u0 = np.exp(-((x)**2.)/(sigma**2.))
"""
sigma = L/4.
u0 = np.exp(-((x - L/2.)/sigma)**2.)
#u0 = np.exp(-((x)/sigma)**2.)
#u0 = np.sin(x)

x2 = np.linspace(0, 2*L*(1. - (1./(2.*N))), N)
u0 = np.exp(-((x2 - L/2.)**2.)/(sigma**2.)) - np.exp(-((x2 - 3.*L/2.)**2.)/(sigma**2.))  # Dirichletov robni
#u0 = np.sin(np.pi*x)
u0hat = np.fft.fft(u0)
x = x2
# SciPy's odeint function doesn't play well with complex numbers so we are
# the state u0hat from an N-element complex vector to  a 2N-element real vector
u0hat_ri = np.concatenate((u0hat.real, u0hat.imag))

# Simulate in Fourier frequency domain
dt = 0.01
tk = 20
t = np.arange(0, tk, dt)


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

uhat_ri = odeint(rhsHeat, u0hat_ri, t, args=(kappa, a))

uhat = uhat_ri[:, :N] + (1j) * uhat_ri[:, N:]

u = np.zeros_like(uhat)

for k in range(len(t)):
    u[k, :] = np.fft.ifft(uhat[k, :])
    
u = u.real

# Waterfall plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.set_cmap('jet_r')
u_plot = u[0:-1:10, :]
u_plot = u[::tk, :]     # this is just making the discrete amount of samples
u_plot = u             # Continous plot, adjust colors
"""
for j in range(u_plot.shape[0]):
    ys = j*np.ones(u_plot.shape[1])
    #ax.plot(x, ys, u_plot[j, :], color=cm.jet_r(j*tk*2))
    ind = int(len(x)/2)
    ax.plot(x[:ind], ys[:ind], u_plot[j, :ind], color=cm.jet_r(j*tk*2))    # Dirichletov
"""
# For continous plot this is better

X, T = np.meshgrid(x, t)
ind = int(len(u_plot)/2)
surf = ax.plot_surface(X, T, u_plot[:, :ind], cmap='jet')    # Dirichletov

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$')
ax.set_zlabel(r'T($x, t$)')
ax.set_title(r'T($x, t$)')

# Add colorbar
fig.colorbar(surf, ax=ax, label='Temperature')



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

plt.show()