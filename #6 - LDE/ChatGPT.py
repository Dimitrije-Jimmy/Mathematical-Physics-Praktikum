import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

from diffeq import *


def measure_performance(method, f, x0, t):
    start_time = time.time()
    method(f, x0, t)
    end_time = time.time()

    cpu_percent = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent

    execution_time = end_time - start_time

    print(f"Method Execution Time: {execution_time} seconds")
    print(f"CPU Usage: {cpu_percent}%")
    print(f"RAM Usage: {ram_usage}%")

    #return execution_time, cpu_percent, ram_usage

"""
def measure_error(method, analytical_solution, f, x0, t):
    numerical_solution = method(f, x0, t)

    # Calculate absolute error at each time point
    absolute_error = np.abs(numerical_solution - analytical_solution(t))

    # Print or return the error values as needed
    print(f"Absolute Error for {str(method)}: {absolute_error}")

def measure_error(numerical, analytical):
    # Calculate absolute error at each time point
    absolute_error = np.abs(numerical - analytical(t))

    # Print or return the error values as needed
    print(f"Absolute Error: {absolute_error}")

    return absolute_error
"""

def measure_stability(method, f, x0, t, step_sizes):
    """
    Measure stability of a numerical integration method.

    Parameters:
    - method: The numerical integration method to test.
    - f: The function representing the system of ODEs.
    - x0: The initial condition(s).
    - t: Array of time values.
    - step_sizes: List of different step sizes to test.

    Returns:
    - None (for now, you may modify it as needed)
    """
    cmap = plt.get_cmap('viridis')
    barve = [cmap(value) for value in np.linspace(0, 1, len(step_sizes))]

    plt.figure(figsize=(10, 6))

    for i, h in enumerate(step_sizes):
        a, b = ( 0.0, 20.0 )
        n = int((b-a)/abs(h))
        t = np.linspace( a, b, n )
        x = method(f, x0, t)

        # Plot the solution for each step size
        plt.plot(t, x, c=barve[i], label=f'Step Size: {h}')

    x_analyt = T_out + np.exp(-k*t)*(x0 - T_out)
    plt.plot(t, x_analyt, 'k-o', label='analytical')

    plt.title('Stability Test')
    plt.xlabel('Time')
    plt.ylabel('Solution')
    plt.legend()
    plt.show()



def f( x, t, k=0.1, T_out = -5 ):
#        return x * numpy.sin( t )
    #return t * numpy.sin( t )
    return -k*(x - T_out)

a, b = ( 0.0, 20.0 )
#x0 = 1.0
x0 = 21

T_out = -5
k = 0.1

h = 2
n = int((b-a)/np.abs(h))
t = np.linspace( a, b, n )

# compute various numerical solutions
x_euler = euler( f, x0, t )
x_heun = heun( f, x0, t )
x_rk2 = rk2a( f, x0, t )
x_rk4 = rku4( f, x0, t )
x_pc4 = pc4( f, x0, t )
t_rkf, x_rkf = rkf( f, a, b, x0, 1e-6, 1.0, 0.01 ) # unequally spaced t

#data = solve_ivp( f, (a, b), [x0] )
#t_ivp, x_ivp = data[0], data[1]
x_ode = odeint( f, x0, t )


# compute true solution values in equal spaced and unequally spaced cases
#x = sin(t) - t*cos(t)
#xrkf = sin(t_rkf) - t_rkf*cos(t_rkf)
x_analyt = T_out + np.exp(-k*t)*(x0 - T_out)
xrkf = T_out + np.exp(-k*t_rkf)*(x0 - T_out)


# Example usage:
#measure_performance(euler, f, x0, t)

#measure_error(x_euler, x_analyt)

plt.figure(figsize=(12, 6))

plt.subplot( 1, 2, 1 )
plt.plot(t, x_analyt, 'k-o', alpha=0.8, lw=0.8, label='analytical')
plt.plot(t, x_euler, 'b-o', alpha=0.8, lw=0.8, label='euler')
plt.plot(t, x_heun, 'g-o', alpha=0.8, lw=0.8, label='heun')
plt.plot(t, x_rk2, 'r-o', alpha=0.8, lw=0.8, label='rk2')
plt.plot(t, x_rk4, 'c-o', alpha=0.8, lw=0.8, label='rk4')
#plt.plot(t, x_pc4, 'y-o', alpha=0.8, lw=0.8, label='pc4')
#plt.plot(#t_ivp, x_ivp, 'w-o', )
#plt.plot(#t, x_ode, 'w-o', )# )


plt.legend()

plt.subplot( 1, 2, 2 )
plt.plot(t, np.abs(x_euler - x_analyt), 'b-o', alpha=0.8, lw=0.8, label='euler')
plt.plot(t, np.abs(x_heun - x_analyt), 'g-o', alpha=0.8, lw=0.8, label='heun')
plt.plot(t, np.abs(x_rk2 - x_analyt), 'r-o', alpha=0.8, lw=0.8, label='rk2')
plt.plot(t, np.abs(x_rk4 - x_analyt), 'c-o', alpha=0.8, lw=0.8, label='rk4')
#plt.plot(t, np.abs(x_pc4 - x_analyt), 'y-o', alpha=0.8, lw=0.8, label='pc4')
#t_ivp, np.abs(x_ivp - x_analyt), 'w-o',
#t, np.abs(x_ode - x_analyt), 'w-o',)
plt.legend()

plt.tight_layout()
plt.show()

step_sizes = np.linspace(-3, 0, 5, endpoint=False)
#measure_stability(euler, f, x0, t, step_sizes)



import sys
sys.exit()
#figure( 1 )
subplot( 2, 2, 1 )
plot( t, x_euler, 'b-o', t, x_heun, 'g-o', t, x_rk2, 'r-o' )
xlabel( '$t$' )
ylabel( '$x$' )
title( 'Solutions of $dx/dt = t\,\sin(t) \sin(t)$, $x(0)=1$' )
legend( ( 'Euler', 'Heun', 'Midpoint' ), loc='lower left' )

#figure( 2 )
subplot( 2, 2, 2 )
plot( t, x_euler - x, 'b-o', t, x_heun - x, 'g-o', t, x_rk2 - x, 'r-o' )
xlabel( '$t$' )
ylabel( '$x - x^*$' )
title( 'Errors in solutions of $dx/dt = t\, \sin(t)$, $x(0)=1$' )
legend( ( 'Euler', 'Heun', 'Midpoint' ), loc='upper left' )

#figure( 3 )
subplot( 2, 2, 3 )
plot( t, x_rk4, 'b-o', t, x_pc4, 'g-o', t_rkf, x_rkf, 'r-o' )
xlabel( '$t$' )
ylabel( '$x$' )
title( 'Solutions of $dx/dt = t\, \sin(t)$, $x(0)=1$' )
legend( ( '$O(h^4)$ Runge-Kutta', '$O(h^4)$ Predictor-Corrector', \
          'Runge-Kutta-Fehlberg' ), loc='lower left' )

#figure( 4 )
subplot( 2, 2, 4 )
plot( t, x_rk4 - x, 'b-o', t, x_pc4 - x, 'g-o', t_rkf, x_rkf - xrkf, 'r-o' )
xlabel( '$t$' )
ylabel( '$x - x^*$' )
title( 'Errors in solutions of $dx/dt = t\, \sin(t)$, $x(0)=1$' )
legend( ( '$O(h^4)$ Runge-Kutta', '$O(h^4)$ Predictor-Corrector', \
          'Runge-Kutta-Fehlberg' ), loc='lower left' )

tight_layout()
show()
