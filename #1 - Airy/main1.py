import numpy as np
import matplotlib.pyplot as plt

import decimal as dc
"""
from decimal import *
getcontext().prec = 6
Decimal(1) / Decimal(7)
Decimal('0.142857')
getcontext().prec = 28
Decimal(1) / Decimal(7)
Decimal('0.1428571428571428571428571429')


from decimal import *
getcontext()
Context(prec=28, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999,
        capitals=1, clamp=0, flags=[], traps=[Overflow, DivisionByZero,
        InvalidOperation])

getcontext().prec = 7       # Set a new precision
"""


import mpmath as mp
"""
from mpmath import sin, cos
sin(1), cos(1)

import mpmath
mpmath.sin(1), mpmath.cos(1)

from mpmath import mp    # mp context object -- to be explained
mp.sin(1), mp.cos(1)

mpf(3) + 2*mpf('2.5') + 1.0
mpf('9.0')
mp.dps = 15      # Set precision (see below)
mpc(1j)**0.5
mpc(real='0.70710678118654757', imag='0.70710678118654757')
"""

#x = np.linspace(-15, 5, 1000)

"""
Diferencialna za rešit:     y''(x) - xy(x) = 0

Ai(x) = 1/\pi \int_0^\infty \cos(t^3/3 + xt) dt
Bi(x) = 1/\pi \int_0^\infty [e^(-t^3/3 + xt) + \sin(t^3/3 + xt)] dt



"""
"""
# example from library scipy.special.airy
from scipy import special
x = np.linspace(-15, 5, 201)
ai, aip, bi, bip = special.airy(x)

plt.plot(x, ai, 'r', label='Ai(x)')
plt.plot(x, bi, 'b--', label='Bi(x)')
plt.ylim(-0.5, 1.0)
plt.grid()
plt.legend(loc='upper left')
#plt.show()
plt.clf()
"""

# Solving the DE
"""
We define the differential equation as a Python function diff_eq(x, y) where y is a vector containing y and y' (the first derivative of y).

We specify the range of x values, the step size, and initialize arrays to store the y and y' values.

We use a finite difference method (the fourth-order Runge-Kutta method) to numerically solve the differential equation for each x value in the range.

Finally, we plot the numerical solution for y(x) and y'(x).

You can adjust the initial conditions, range, and step size as needed for your specific problem.
"""

from tqdm import tqdm  # Import tqdm for the progress bar
"""
# Define the differential equation as a function
def diff_eq(x, y):
    return y[1], x * y[0]

# Define the range and step size for x values
x_min = -10
x_max = 10
step_size = 0.01
x_values = np.arange(x_min, x_max + step_size, step_size)

# Initialize the y values and derivatives
y_values = np.zeros((2, len(x_values)))
y_values[0, 0] = 1.0  # Initial condition y(0) = 1
y_prime_values = np.zeros_like(x_values)

# Create a tqdm progress bar
progress_bar = tqdm(total=len(x_values) - 1, desc="Solving Differential Equation")

# Numerically solve the differential equation using finite differences
for i in range(len(x_values) - 1):
    h = x_values[i + 1] - x_values[i]
    k1 = h * np.array(diff_eq(x_values[i], y_values[:, i]))
    k2 = h * np.array(diff_eq(x_values[i] + h / 2, y_values[:, i] + k1 / 2))
    k3 = h * np.array(diff_eq(x_values[i] + h / 2, y_values[:, i] + k2 / 2))
    k4 = h * np.array(diff_eq(x_values[i] + h, y_values[:, i] + k3))
    y_values[:, i + 1] = y_values[:, i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    y_prime_values[i + 1] = y_values[1, i + 1]

    # Update the progress bar
    progress_bar.update(1)

# Close the progress bar
progress_bar.close()

# Plot the solution
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values[0, :], label='y(x)', linewidth=2)
plt.plot(x_values, y_prime_values, label="y'(x)", linewidth=2)
plt.xlabel('x')
plt.ylabel('Values')
plt.legend()
plt.title('Numerical Solution to d^2y/dx^2 - xy = 0')
plt.grid(True)
plt.show()
"""
"""
from scipy.integrate import solve_ivp

# Define the Airy differential equation as a function
def airy_eq(x, y):
    return [y[1], x * y[0]]

# Define the range for x values
x_min = -15
x_max = 5

# Set the initial conditions for y and y'
initial_conditions = [1.0, 0.0]  # y(0) = 1, y'(0) = 0

# Create a list of x values for which to compute the solution
x_values = np.linspace(x_min, x_max, 400)

# Solve the Airy differential equation using solve_ivp
solution = solve_ivp(airy_eq, [x_min, x_max], initial_conditions, t_eval=x_values)

# Extract y(x) and y'(x) from the solution
y_values = solution.y[0]
y_prime_values = solution.y[1]

# Plot the solution
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label='y(x)', linewidth=2)
plt.plot(x_values, y_prime_values, label="y'(x)", linewidth=2)
plt.xlabel('x')
plt.ylabel('Values')
plt.legend()
plt.title('Numerical Solution to Airy Differential Equation using solve_ivp')
plt.grid(True)
plt.show()
"""
"""
import scipy.integrate as spi

# Define the integrand for Ai(x)
def integrand_ai(t, x):
    return np.cos(t**3 / 3 + x * t)

# Define the integrand for Bi(x)
def integrand_bi(t, x):
    return np.exp(-t**3 / 3 + x * t) + np.sin(t**3 / 3 + x * t)

# Define the range of x values
x_values = np.linspace(-15, 5, 400)

# Initialize arrays to store the results
integral_ai_values = np.zeros_like(x_values)
integral_bi_values = np.zeros_like(x_values)

# Compute the integrals for Ai(x) and Bi(x) for each x value
for i, x in enumerate(x_values):
    integral_ai, _ = spi.quad(integrand_ai, 0, np.inf, args=(x,))
    integral_bi, _ = spi.quad(integrand_bi, 0, np.inf, args=(x,))
    integral_ai_values[i] = integral_ai #/ np.pi  # Normalize by 1/pi
    integral_bi_values[i] = integral_bi #/ np.pi  # Normalize by 1/pi

# Normalize the results to match scipy.special.airy
integral_ai_values /= np.pi
integral_bi_values /= np.pi


# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(x_values, integral_ai_values, label='Integral of Ai(x)', linewidth=2)
plt.plot(x_values, integral_bi_values, label='Integral of Bi(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('Values')
plt.legend()
plt.title('Numerical Integration of Airy Functions')
plt.grid(True)
plt.show()

# Doesn't even compute anywhere near to the correct thing
"""




import numpy as np
from scipy.special import airy
import math

def Airy_Ai(x, cutoff=5.0):
    if abs(x) < cutoff:
        # Use Maclaurin series
        s = 0.0
        for n in range(50): # Use 50 terms; adjust as needed
            term = ((-1)**n * math.factorial(3*n)) / (math.factorial(n)**3 * math.factorial(3*n + 1)) * x**(3*n)
            s += term
        return s / np.pi
    elif x > 0:
        # Use asymptotic series for x > 0
        s = 0.0
        for n in range(50): # Use 50 terms; adjust as needed
            term = ((-1)**n * math.factorial(2*n)) / (math.factorial(2*n + 2) * (2*x**(3/2))**(2*n))
            s += term
        return (math.exp(-2/3 * x**(3/2)) / (2*math.sqrt(np.pi) * x**(1/4))) * s
    else:
        # Use asymptotic series for x < 0
        s = 0.0
        for n in range(50): # Use 50 terms; adjust as needed
            term = (math.factorial(2*n)) / (math.factorial(2*n + 2) * (2*abs(x)**(3/2))**(2*n))
            s += term
        return (1/(math.sqrt(np.pi) * abs(x)**(1/4))) * math.sin(2/3 * abs(x)**(3/2) + np.pi/4) * s

# Test the implementation
x_values = np.linspace(-10, 10, 400)
my_airy_values = [Airy_Ai(x) for x in x_values]
true_airy_values = airy(x_values)[0]

error = np.abs(my_airy_values - true_airy_values)

print(f"Maximum Error: {np.max(error)}")

# You can also plot the values and the errors if needed
"""
Remember:

The choice of the cutoff and the number of terms in each series can be adjusted to optimize accuracy and performance.
As with all numerical methods, there's a tradeoff between accuracy and computational cost. 
Depending on your needs, you may choose to use more or fewer terms in the series, or to adjust the cutoff.
"""



# Recursive implementation
import numpy as np
from scipy.special import airy
import math

def maclaurin_term(x, n, prev_term):
    if n == 0:
        return x**0 / np.pi  # The initial term
    multiplier = (-x**3 * (3*n-2) * (3*n-1) * (3*n)) / ((n**3) * (3*n-1) * (3*n))
    return prev_term * multiplier

def asymptotic_term(x, n, prev_term, is_positive_x):
    if n == 0:
        return 1.0  # The initial term
    if is_positive_x:
        multiplier = - (2*n-1) / (2*x**(3/2) * (2*n))
    else:
        multiplier = (2*n-1) / (2*abs(x)**(3/2) * (2*n))
    return prev_term * multiplier

def Airy_Ai_recursive(x, cutoff=5.0, max_terms=50):
    if abs(x) < cutoff:
        s, term = 0.0, 1.0
        for n in range(max_terms):
            term = maclaurin_term(x, n, term)
            s += term
        return s
    elif x > 0:
        s, term = 0.0, 1.0
        for n in range(max_terms):
            term = asymptotic_term(x, n, term, True)
            s += term
        return (math.exp(-2/3 * x**(3/2)) / (2*math.sqrt(np.pi) * x**(1/4))) * s
    else:
        s, term = 0.0, 1.0
        for n in range(max_terms):
            term = asymptotic_term(x, n, term, False)
            s += term
        return (1/(math.sqrt(np.pi) * abs(x)**(1/4))) * math.sin(2/3 * abs(x)**(3/2) + np.pi/4) * s

# Test the implementation
x_values = np.linspace(-10, 10, 400)
my_airy_values = [Airy_Ai_recursive(x) for x in x_values]
true_airy_values = airy(x_values)[0]

error = np.abs(my_airy_values - true_airy_values)

print(f"Maximum Error: {np.max(error)}")

"""
In the above code:

maclaurin_term computes the nth term of the Maclaurin series based on the previous term.

asymptotic_term computes the nth term of the asymptotic series based on the previous term, 
 taking into account whether x is positive or negative.
Airy_Ai_recursive uses these helper functions to compute the Airy function using either 
 the Maclaurin or the asymptotic series, based on the magnitude of x.

Note: This approach isn't fully recursive in the sense of a function calling itself. 
 It uses recursion in the computation of terms within the series.
"""
