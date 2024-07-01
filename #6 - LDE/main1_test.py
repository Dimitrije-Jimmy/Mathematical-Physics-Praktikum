import numpy as np
import matplotlib.pyplot as plt
import time
import psutil

from diffeq import *


def f( x, t, k=0.1, T_out = -5 ):
#        return x * numpy.sin( t )
    #return t * numpy.sin( t )
    return -k*(x - T_out)

a, b = ( 0.0, 31.0 )
#x0 = 1.0
x0 = 21.0

T_out = -5.0
k = 0.1

#h = 2
#n = int((b-a)/np.abs(h))

#h2 = 2
#h1 = 1
#h05 = 0.5
#h01 = 0.1
#h001 = 0.001
#h105 = 1e-5
#t2 = np.arange( a, b, h2 )
#t1 = np.arange( a, b, h1 )
#t05 = np.arange( a, b, h05 )
#t01 = np.arange( a, b, h01 )
#t001 = np.arange( a, b, h001 )
#t105 = np.arange( a, b, h105)

# compute various numerical solutions
#x1 = euler( f, x0, t1 )

t = np.linspace( a, b, 10 )
x_analyt = T_out + np.exp(-k*t)*(x0 - T_out)

h_sez = [2.0, 1.0, 0.5, 0.1, 0.001, 1e-4]
#h_sez = np.linspace(1e-4, 2.0, 100)[::-1]

# barve
cmap = plt.get_cmap('viridis_r')
cmap2 = plt.get_cmap('magma_r')
barve = [cmap(value) for value in np.linspace(0, 1, len(h_sez))]
barve2 = [cmap2(value) for value in np.linspace(0, 1, len(h_sez))]

plt.figure(figsize=(12, 5))

plt.subplot( 1, 2, 1 )
plt.plot(t, x_analyt, c='k', ls='--', alpha=0.8, lw=0.8, label='analytical')

for i, h in enumerate(h_sez[::-1]):
    print(i, h)
    th = np.arange( a, b, h )
    xh, err = rk45( f, x0, th )
    xh2, err2 = rk45( f, -15.0, th )
    #xh = rk2a( f, x0, th )
    #xh2 = rk2a( f, -15.0, th )

    plt.subplot( 1, 2, 1 )
    plt.plot(th, xh, ls='-', marker='.', c=barve[i], alpha=(1-h*0.1), lw=(1-h*0.1), label=f'h = {h}')
    
    # se za x0=-15
    plt.plot(th, xh2, ls='-', marker='.', c=barve2[i], alpha=(1-h*0.1), lw=(1-h*0.1))#, label=f'h = {h}')
    #plt.yscale('log')
    plt.xlabel(r'time t [s]')
    plt.ylabel(r'T [$^{\circ}$C]')
    plt.title(r'Rezultati')
    plt.legend()

    xh_analyt = T_out + np.exp(-k*th)*(x0 - T_out)
    plt.subplot( 1, 2, 2 )
    plt.plot(th, np.abs(xh - xh_analyt), ls='-', marker='o', c=barve[i], alpha=(1-h*0.1), lw=(1-h*0.1), label=f'h = {h}')
    #plt.yscale('log')
    plt.xlabel(r'time t [s]')
    plt.ylabel(r'$\Delta\varepsilon$')
    plt.title(r'Absolutna napaka')
    plt.legend()


plt.tight_layout()
plt.show()