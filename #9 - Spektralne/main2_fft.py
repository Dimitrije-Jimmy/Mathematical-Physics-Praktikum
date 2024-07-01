import numpy as np
import matplotlib.pyplot as plt
import time
import os

from scipy import integrate

from diffeq import *
from main1_analytic2 import *


a, b, n, m = 0., 1., 2, 100
L, D = 1.0, 0.05
sigma = L/4.
x0 = np.linspace(a, b, m)
t0 = np.linspace(0., 1., m*n)

zp = np.exp(-((x0 - L/2.)**2.)/(sigma**2.))

N, M, P = n, m, 100

#_______________________________________________________________________________
"""
\frac{\partial T_k(t)}{\partial t} = D(-2\pi fk)^2 T_k(t)

integracija:
T_k(t + h) = T_k(t) + h*D(-2\pi fk)^2 T_k(t)

analitično:
T_k(t) = C*exp(-4pi^2 fk^2 D t)

T(x, t) = ifft (T_k(t))

fk = k/a, 0<x<a

"""

#   Tk(t) = C*exp(-4pi^2 fk^2 D t)
#   C dobimo iz začetnega pogoja (FFT Gaussa) vsako Tk(t) 
# mormao pomnožiti s svojim C-jem
#   -f_N < fk < f_N

h_time = t0[1] - t0[0]
k = np.linspace(-(m*n-2)/2, (m*n)/2, m*n)**2

x1 = np.linspace(0, L*(1. - (1./m)), m)
x2 = np.linspace(0, 2*L*(1. - (1./(2.*m))), 2*m)
x3 = np.linspace(-L, 2*L*(1. - (1./(3.*m))), 3*m)
zp1 = np.exp(-((x1 - L/2.)**2.)/(sigma**2.))
zp2 = np.exp(-((x2 - L/2.)**2.)/(sigma**2.)) - np.exp(-((x2 - 3.*L/2.)**2.)/(sigma**2.))


x = x2
zp = zp2
m = 2

fft = np.fft.fft(zp)
C = np.fft.fftshift(fft) #MORA BITI KOMPLEKSNO

Tkt1 = C * np.exp(-4.*(np.pi**2.)*D*np.outer(t0, k)/(L**2) )

Tkt = Tkt1

plt.figure()            # FIGURE 3
plt.subplot(1, 2, 1)
plt.imshow(numpy.real(Tkt), cmap='jet', interpolation='nearest')
plt.subplot(1, 2, 2)
plt.imshow(numpy.imag(Tkt), cmap='jet', interpolation='nearest')
#vzdolž vrstice so isti t-ji
#IFFT moramo narediti na vrsticah! axis=1


Uxt = np.fft.ifft(np.fft.ifftshift(Tkt1, axes=1), axis=1)


plt.figure()            # FIGURE 4
plt.subplot(1, 2, 1)
plt.imshow(numpy.real(Uxt), cmap='jet', interpolation='nearest')
plt.xlabel('$x$')
plt.ylabel('$t$')

plt.subplot(1, 2, 2)
plt.imshow(numpy.real(Uxt[:,:int(m*n/2)]), cmap='jet', interpolation='nearest')
plt.xlabel('$x$')
plt.ylabel('$t$')

colors = plt.cm.jet(numpy.linspace(1,0,m))
plt.figure()            # FIGURE 5
plt.subplot(1, 2, 1)
plt.xlabel('$x$')
plt.ylabel('$T$')
for i in range(m):
    plt.plot(numpy.real(Uxt[i]), color=colors[i])

plt.subplot(1, 2, 2)
plt.xlabel('$x$')
plt.ylabel('$T$')
for i in range(m):
    plt.plot(numpy.real(Uxt[i,:int(m*n/2)]), color=colors[i])

plt.show()