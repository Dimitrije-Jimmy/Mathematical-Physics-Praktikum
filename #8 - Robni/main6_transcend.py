import numpy as np
import matplotlib.pyplot as plt
#plt.style.use(['science', 'notebook', 'dark_background'])
plt.style.use('dark_background')
from scipy.linalg import eigh_tridiagonal

from bvp import *

# Functions ____________________________________________________________________

def rk4( f, x0, t, V, E ):
    """Fourth-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = rku4(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.
    """

    n = len( t )
    x = numpy.array( [ x0 ] * n )
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        k1 = h * f( x[i], t[i], V, E )
        k2 = h * f( x[i] + 0.5 * k1, t[i] + 0.5 * h, V, E )
        k3 = h * f( x[i] + 0.5 * k2, t[i] + 0.5 * h, V, E )
        k4 = h * f( x[i] + k3, t[i+1], V, E )
        x[i+1] = x[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0

    return x


def shoot( f, a, b, z1, z2, t, tol, V, E ):
    """Implements the shooting method to solve second order BVPs

    USAGE:
        y = shoot(f, a, b, z1, z2, t, tol)

    INPUT:
        f     - function dy/dt = f(y,t).  Since we are solving a second-
                order boundary-value problem that has been transformed
                into a first order system, this function should return a
                1x2 array with the first entry equal to y and the second
                entry equal to y'.
        a     - solution value at the left boundary: a = y(t[0]).
        b     - solution value at the right boundary: b = y(t[n-1]).
        z1    - first initial estimate of y'(t[0]).
        z1    - second initial estimate of y'(t[0]).
        t     - array of n time values to determine y at.
        tol   - allowable tolerance on right boundary: | b - y[n-1] | < tol

    OUTPUT:
        y     - array of solution function values corresponding to the
                values in the supplied array t.

    NOTE:
        This function assumes that the second order BVP has been converted to
        a first order system of two equations.  The secant method is used to
        refine the initial values of y' used for the initial value problems.
    """

    #from diffeq import rk4
    #from diffeq import rku4 as rk4

    max_iter = 25   # Maximum number of shooting iterations

    n = len( t )    # Determine the size of the arrays we will generate

    # Compute solution to first initial value problem (IVP) with y'(a) = z1.
    # Because we are using the secant method to refine our estimates of z =
    # y', we don't really need all the solution of the IVP, just the last
    # point of it -- this is saved in w1.

    y = rk4( f, [a,z1], t, V, E )
    w1 = y[n-1,0]

    print ("%2d: z = %10.3e, error = %10.3e" % ( 0, z1, b - w1 ))

    # Begin the main loop.  We will compute the solution of a second IVP and
    # then use the both solutions to refine our estimate of y'(a).  This
    # second solution then replaces the first and a new "second" solution is
    # generated.  This process continues until we either solve the problem to
    # within the specified tolerance or we exceed the maximum number of
    # allowable iterations.

    for i in range( max_iter ):

        # Solve second initial value problem, using y'(a) = z2.  We need to
        # retain the entire solution vector y since if y(t(n)) is close enough
        # to b for us to stop then the first column of y becomes our solution
        # vector.

        y = rk4( f, [a,z2], t, V, E )
        w2 = y[n-1,0]

        print ("%2d: z = %10.3e, error = %10.3e" % ( i+1, z2, b - w2 ))

        # Check to see if we are done...

        if abs( b - w2 ) < tol:
            break

        # Compute the new approximations to the initial value of the first
        # derivative.  We compute z2 using a linear fit through (z1,w1) and
        # (z2,w2) where w1 and w2 are the estimates at t=b of the initial
        # value problems solved above with y1'(a) = z1 and y2'(a) = z2.  The
        # new value for z1 is the old value of z2.

        #z1, z2 = ( z2, z1 + ( z2 - z1 ) / ( w2 - w1 ) * ( b - w1 ) )
        z1, z2 = ( z2, z2 + ( z2 - z1 ) / ( w2 - w1 ) * ( b - w2 ) )
        w1 = w2

    # All done.  Check to see if we really solved the problem, and then return
    # the solution.

    if abs( b - w2 ) >= tol:
        print ("\a**** ERROR ****")
        print ("Maximum number of iterations (%d) exceeded" % max_iter)
        print ("Returned values may not have desired accuracy")
        print ("Error estimate of returned solution is %e" % ( b - w2 ))

    return y[:,0]

def shoot2( f, a, b, z1, z2, t, tol ):
    """Implements the shooting method to solve second order BVPs

    USAGE:
        y = shoot(f, a, b, z1, z2, t, tol)

    INPUT:
        f     - function dy/dt = f(y,t).  Since we are solving a second-
                order boundary-value problem that has been transformed
                into a first order system, this function should return a
                1x2 array with the first entry equal to y and the second
                entry equal to y'.
        a     - solution value at the left boundary: a = y(t[0]).
        b     - solution value at the right boundary: b = y(t[n-1]).
        z1    - first initial estimate of y'(t[0]).
        z1    - second initial estimate of y'(t[0]).
        t     - array of n time values to determine y at.
        tol   - allowable tolerance on right boundary: | b - y[n-1] | < tol

    OUTPUT:
        y     - array of solution function values corresponding to the
                values in the supplied array t.

    NOTE:
        This function assumes that the second order BVP has been converted to
        a first order system of two equations.  The secant method is used to
        refine the initial values of y' used for the initial value problems.
    """

    #from diffeq import rk4
    from diffeq import rku4 as rk4

    max_iter = 25   # Maximum number of shooting iterations

    n = len( t )    # Determine the size of the arrays we will generate

    # Compute solution to first initial value problem (IVP) with y'(a) = z1.
    # Because we are using the secant method to refine our estimates of z =
    # y', we don't really need all the solution of the IVP, just the last
    # point of it -- this is saved in w1.

    y = rk4( f, [a,z1], t )
    w1 = y[n-1,0]

    print ("%2d: z = %10.3e, error = %10.3e" % ( 0, z1, b - w1 ))

    # Begin the main loop.  We will compute the solution of a second IVP and
    # then use the both solutions to refine our estimate of y'(a).  This
    # second solution then replaces the first and a new "second" solution is
    # generated.  This process continues until we either solve the problem to
    # within the specified tolerance or we exceed the maximum number of
    # allowable iterations.

    for i in range( max_iter ):

        # Solve second initial value problem, using y'(a) = z2.  We need to
        # retain the entire solution vector y since if y(t(n)) is close enough
        # to b for us to stop then the first column of y becomes our solution
        # vector.

        y = rk4( f, [a,z2], t )
        w2 = y[n-1,0]

        print ("%2d: z = %10.3e, error = %10.3e" % ( i+1, z2, b - w2 ))

        # Check to see if we are done...

        if abs( b - w2 ) < tol:
            break

        # Compute the new approximations to the initial value of the first
        # derivative.  We compute z2 using a linear fit through (z1,w1) and
        # (z2,w2) where w1 and w2 are the estimates at t=b of the initial
        # value problems solved above with y1'(a) = z1 and y2'(a) = z2.  The
        # new value for z1 is the old value of z2.

        #z1, z2 = ( z2, z1 + ( z2 - z1 ) / ( w2 - w1 ) * ( b - w1 ) )
        z1, z2 = ( z2, z2 + ( z2 - z1 ) / ( w2 - w1 ) * ( b - w2 ) )
        w1 = w2

    # All done.  Check to see if we really solved the problem, and then return
    # the solution.

    if abs( b - w2 ) >= tol:
        print ("\a**** ERROR ****")
        print ("Maximum number of iterations (%d) exceeded" % max_iter)
        print ("Returned values may not have desired accuracy")
        print ("Error estimate of returned solution is %e" % ( b - w2 ))

    return y[:,0]


def schrodinger(y, r, V, E):
    dydt = [y[1], (- E) * y[0]]
    return np.asarray(dydt)

def analytic2(x, k, A=5.0):
    
    if k % 2 == 0:
        return np.sin(k/(2*A)*np.pi*x)*np.sqrt(2/np.pi)
    else:
        return np.cos(k/(2*A)*np.pi*x)*np.sqrt(2/np.pi)
    
    
def analytic(x): 
    return np.sqrt(2/np.pi)*np.sin(n*x)
    
# Iskanje Energij ______________________________________________________________

V0 = 100
#V0 = 0
E = np.linspace(-V0, 0, 1000000)

RHS = np.sqrt(-E)
LHS1 = np.sqrt(E+V0)*np.tan(np.sqrt(E+V0))
LHS2 = -np.sqrt(E+V0)/np.tan(np.sqrt(E+V0+1e-9))


plt.figure(figsize=(8,5))
plt.scatter(E, LHS1, s=1, color='blue', label=r'LHS: $\sqrt{E+V}\tan{\sqrt{E+V}}$')
plt.scatter(E, LHS2, s=1, color='blue', label=r'LHS: $-\sqrt{E+V}\cot{\sqrt{E+V}}$')
plt.scatter(E, RHS, s=1, color='red', label=r'RHS: $\sqrt{-E}$')
plt.ylim(-20,20)
plt.grid(color='grey')
plt.xlabel('$E$')
plt.ylabel('LHS in RHS')
plt.legend()
#plt.show()
plt.clf()
plt.close()

def f1(E, V0):
    return np.sqrt(E+V0)*np.tan(np.sqrt(E+V0)) - np.sqrt(-E)
def f2(E, V0, eps=1e-10):
    return np.sqrt(E+V0+eps)/np.tan(np.sqrt(E+V0+eps)) + np.sqrt(-E)


plt.figure(figsize=(8,5))
plt.scatter(E, f1(E,V0), s=1, color='blue')
plt.scatter(E, f2(E,V0), s=1, color='blue')
plt.grid(color='grey')
plt.ylim(-20,20)
plt.xlabel('$E$')
plt.ylabel('razlika LHS in RHS')
#plt.show()
plt.clf()
plt.close()

f1s = f1(E, V0)
f2s = f2(E, V0)

zero_crossings_even = np.where(np.diff(np.sign(f1s)) * (np.abs(f1s[:-1])<3).astype(float))[0]
zero_crossings_odd = np.where(np.diff(np.sign(f2s)) * (np.abs(f2s[:-1])<3).astype(float))[0]
zero_crossings = np.sort(np.concatenate([zero_crossings_even, zero_crossings_odd]))
Es_method1 =  (E[zero_crossings] + E[zero_crossings+1])/2

print(Es_method1+V0)
"""
[0.01223866 0.04895417 0.11014508 0.19580898 0.30594251 0.44054134
 0.59960017 0.78311275 0.99107186 1.22346933]

[ 2.03520352  8.13581358 18.24682468 32.25822582 49.969997   70.95209521
 93.68436844]
"""

plt.style.use('default')

V = V0
a = -1.0
b = 1.0
N = 200
x = np.linspace( a, b, N )
eigE = np.linspace(0, V0, N)
eigPsi = np.array([])
"""
for E in eigE:
    n = np.sqrt(E)


    psi0 = shoot( schrodinger, 0., 0., 1., 1.5, x, 1e-5, V, E )
    #psisNorm = psi0 / (np.sqrt(np.sum(psi0**2)*(b-a)/N))
    psisNorm = psi0 / np.max(psi0)
    eigPsi = np.append(eigPsi, psisNorm[-1])
"""

"""
eigPsi = shoot( schrodinger, 0., 0., 1., 1., x, 1e-5, V, E )

fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))

stvr = 1
amplitude = 5
#ax1.plot(x, V_fpw, 'k--', lw=1, alpha=0.8)
ax1.plot([-a, -1], [V, V], c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([1, a], [V, V], c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([-1, 1], [0, 0], c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([-1, -1], [0, V], c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([1, 1], [0, V], c='k', ls='--', lw=0.8, alpha=0.8)
for i in range(stvr):
    #ax1.plot(x, 50 * eigPsi[:, i], label=f"$\\psi_1$, $E_1={np.around(eigE[i], 2)}$")
    vektorji = amplitude*eigPsi + i*V/stvr
    ax1.plot(x, vektorji, label=f"$\\psi_{i}$, $E_{i}={np.around(eigE[i], 2)}$")
    
    
    #ax1.plot(x, analytic_finite(x, i+1, a), label=f"$\\psi_{i}$, analytic")

    # Fill the area under the curve with a solid color (e.g., light blue)
    ax1.fill_between(x, vektorji, y2=i*V/stvr, alpha=0.5)
    
    # absolutne napake ni ker nimamo analiticne resitve...
    #err = np.abs(eigPsi[:, i] - analytic(x, i+1, a))
    #ax2.plot(x, err, label=f"$\\psi_{i+1}$")

ax1.set_title("Vezana stanja končne pot. jame globine 100 a.u z FD")
ax1.set_xlabel("x")
ax1.set_ylabel(r"$\psi_n(x)$")
ax1.legend(loc="upper right")
ax1.set_xlim(-2, 2)

#ax2.set_title(f"Absolutne napake rešitev z FD in $n = {dim}$")
#ax2.set_xlabel("x")
#ax2.set_ylabel(r"$|num - analytic|$")
#ax2.set_yscale("log")
#ax2.legend()

plt.tight_layout()
plt.show()
"""


a = 0.0
b = np.pi
N = 1000
x = np.linspace( a, b, N )

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

stvr = 5
amplitude = 5

cmap = plt.get_cmap('viridis')
cmap2 = plt.get_cmap('magma_r')
barve = [cmap(value) for value in np.linspace(0, 1, stvr)]

for n in range(stvr):
    n += 1
    def f(psi,t, E=n**2):
        return np.array( [psi[1], -E*psi[0]] ) 
    
    eigPsi = shoot2( f, 0.0, 0.0, n*np.sqrt(2/np.pi), n*np.sqrt(2/np.pi), x, 1e-5 )
    #eigPsi = shoot2( f, 0.0, 0.0, 1, 1, x, 1e-5 )
    
    vektor = amplitude*eigPsi + (n-1)*V/stvr
    ax1.plot(x-np.pi/2, vektor, color=barve[n-1], label=f"$\\psi_{n}$, $E_{n} = {n**2}E_1$")
    #ax1.text(1.35, (n-1+0.1)*V/stvr, f'$\\psi_{n}$, $E_{n} = {n**2}E_1$', verticalalignment='center')
    
    ax1.fill(x-np.pi/2, vektor, alpha=0.5, color=barve[n-1])
    
    x_analiticna = analytic( x )
    ax2.plot( x-np.pi/2, np.abs(eigPsi-x_analiticna), color=barve[n-1], label=f"$\\psi_{n}$")#, ls = 'solid' )

ax1.set_title( 'Neskončna potencialna jama' )
ax1.set_ylabel( r'$\Psi_n(x)$' )
ax1.set_xlabel( 'x' )
ax1.legend()
    
    
ax2.set_title( 'Absolutna napaka strelske metode' )
ax2.set_ylabel( r'$\Delta\Psi_n(x)$' )
ax2.set_xlabel( 'x' )
ax2.set_yscale('log')
ax2.legend()

plt.tight_layout()
plt.show()
plt.clf()
plt.close()



fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))

stvr = 5
amplitude = 5
energije = Es_method1+V0

"""
ax1.plot([-a, -1], [V, V], c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([1, a], [V, V], c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([-1, 1], [0, 0], c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([-1, -1], [0, V], c='k', ls='--', lw=0.8, alpha=0.8)
ax1.plot([1, 1], [0, V], c='k', ls='--', lw=0.8, alpha=0.8)
"""
for n in range(stvr):
    n += 1
    def f(psi,t, E=n**2):
        return np.array( [psi[1], -E*psi[0]] ) 
    
    eigPsi = shoot2( f, 0.0, 0.0, n*np.sqrt(2/np.pi), n*np.sqrt(2/np.pi), x, 1e-5 )
    #eigPsi = shoot2( f, 0.0, 0.0, 1, 1, x, 1e-5 )
    
    vektor = amplitude*eigPsi + (n-1)*V/stvr
    ax1.plot(x-np.pi/2, vektor, color=barve[n-1], label=f"$\\psi_{n}$, $E_{n} = {np.around(energije[n-1]/energije[0], 3)}E_1$")
    #ax1.text(1.35, (n-1+0.1)*V/stvr, f'$\\psi_{n}$, $E_{n} = {n**2}E_1$', verticalalignment='center')
    
    ax1.fill(x-np.pi/2, vektor, alpha=0.5, color=barve[n-1])
    
    x_analiticna = analytic( x )
    ax2.plot( x-np.pi/2, np.abs(eigPsi-x_analiticna), color=barve[n-1], label=f"$\\psi_{n}$")#, ls = 'solid' )

ax1.set_title( 'Končna potencialna jama, strelska metoda' )
ax1.set_ylabel( r'$\Psi_n(x)$' )
ax1.set_xlabel( 'x' )
ax1.legend()
    

plt.tight_layout()
plt.show()
    
    