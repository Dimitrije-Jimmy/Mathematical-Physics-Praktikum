import numpy as np
import matplotlib.pyplot as plt
import os
import sys


from diffeq import *
from bvp import *

V = 100
a = -5.0
b = 5.0
n1 = 64
n2 = 128
t1 = np.linspace( a, b, n1 )
t2 = np.linspace( a, b, n2 )

def analytic(x, k, A=5.0):
    #x = np.linspace(-A/2, A/2, N)
    #x = np.linspace(-A/2, A/2, N)
    
    if k % 2 == 0:
        return np.sin(k/(2*A)*np.pi*x)  
    else:
        return np.cos(k/(2*A)*np.pi*x)*2/A

x1 = analytic( t1, 1, b )
x2 = analytic( t2, 1, b )

# Compute finite difference solutions

xfd1 = fd( 0, -V, 0, t1, 0, 0 )
xfd2 = fd( 0, -V, 0, t2, 0, 0 )


plt.plot( t1, xfd1, 'ro', t2, xfd2, 'b-' )
plt.title( 'Finite Difference Method' )
plt.xlabel( '$t$' )
plt.ylabel( '$x$' )
plt.legend( ( '%3d points' % n1, '%3d points' % n2 ), loc='lower right' )
#plt.show()
plt.clf()
plt.close()

# Compute shooting method solutions

def f(x, t, E):
    y, y_odv = x
    dydt = np.array([y_odv, -E*y])
    return dydt

xs1 = shoot( f, 0, 0, 1.0, 0.0, t1, 1e-3 )
xs2 = shoot( f, 0, 0, 1.0, 0.0, t2, 1e-3 )

plt.plot( t1, xs1, 'go', t2, xs2, 'b-' )
plt.title( 'Shooting Method' )
plt.xlabel( '$t$' )
plt.ylabel( '$x$' )
plt.legend( ( '%3d points' % n1, '%3d points' % n2 ), loc='lower right' )

plt.show()