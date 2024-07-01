import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from diffeq import *
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    from pylab import *

    def f( x, t, k=0.1, T_out = -5 ):
#        return x * numpy.sin( t )
        #return t * numpy.sin( t )
        return -k*(x - T_out)

    def analyt( t, x0, k=0.1, T_out=-5.0 ):
        return  T_out + np.exp(-k*t)*(x0 - T_out)

    def integrate(method, f, x0, h):
        n = int((b-a)/np.abs(h))
        t = np.linspace( a, b, n )

        if method == rk45:
            values, _ = method(f, x0, t)
        elif method == rkf:
            t, values = method(f, a, b, x0, 10, 0.3, 1e-6)
        elif method == odeint:
            results, _ = odeint(f, x0, t, full_output=1)
            values = results[:, 0]
        else:
            values = method(f, x0, t)
        return t, values

    a, b = ( 0.0, 100.0 )
    x0 = 21.0
    n=100
    t = numpy.linspace( a, b, n )

    h = 1 # 38 is a nice number, so is 40 ... 
    n=100
    print("h={}".format(h))


    # compute various numerical solutions
    #x_heun = heun( f, x0, t )

    # compute true solution values in equal spaced and unequally spaced cases
    x_true = analyt(t, x0)

    figure(figsize=(10,6))    
    plot( t, x_true, 'k-')
    #for ni in range(10,110,10):
    for ni in range(1,10,1):
        t = numpy.linspace( a, b, ni )
        hi=(b-a)/float(ni)
        x_euler, _ = rk45( f, x0, t )
        plot( t, x_euler, '-o', alpha=0.8, label='h=%.2f'%hi)
    xlabel( r'Čas [$s$]' )
    ylabel( r'T [$^{\circ}$C]' )
    #title( r'Rešitve of $\frac{{\rm d}T}{{\rm d}t} = -0.1\, (T + T_{out}) $, $T(0)=21.0$' )
    title( r'Stabilnost RK45 metode' )
    legend()

    tight_layout()
    show()