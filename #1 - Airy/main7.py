import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
import decimal as dc

# Maclaurian ___________________________________________________________________

# Importing of the reference Scipy Airy functions ______________________________

from scipy.special import airy as scipy_airy

# Define the range of x values
x_values = np.linspace(-15, 10, 400)

# Initialize arrays to store the results
scipy_ai_values, scipy_bi_values = np.zeros_like(x_values), np.zeros_like(x_values)

# Calculate Airy functions using scipy.special.airy and mpmath.airy
for i, x in enumerate(x_values):
    scipy_ai, scipy_aip, scipy_bi, scipy_bip = scipy_airy(x)
    scipy_ai_values[i], scipy_bi_values[i] = scipy_ai, scipy_bi


# Implementing the Maclaurian and Asimptotic expansion _________________________

#alpha, beta = dc.Decimal(0.355028053887817239), dc.Decimal(0.258819403792806798)
epsilon = 1e-11
alpha = 0.355028053887817239  # alpha = Ai(0) = Bi(0)/3^(1/2)
beta = 0.258819403792806798  # beta = Bi'(0)/3^(1/2) = -Ai'(0)


def maclaurian_term_f(x, n, prev_term):
    # just 1 term of the f_n(x)
    if n == 0:
        return 1
    multiplier = (n-2/3) * (x**3) / (n * (3*n-1) * (3*n - 2))
    return multiplier*prev_term

def maclaurian_term_g(x, n, prev_term):
    # just 1 term of the f_n(x)
    if n == 0:
        return x        
    multiplier = (n-1/3) * (x**3) / ((3*n+1) * n * (3*n-1))
    return multiplier*prev_term


def Maclaurian(x, max_terms=500):
    global alpha, beta, epsilon

    sum_f, sum_g, f_term, g_term = 0.0, 0.0, 0.0, 0.0
    for n in range(max_terms):
        f_newterm = maclaurian_term_f(x, n, f_term)
        g_newterm = maclaurian_term_g(x, n, g_term)
        sum_f += f_newterm
        sum_g += g_newterm
        
        print("1")
        if (abs(f_newterm - f_term) <= epsilon) or (abs(g_newterm - g_term) <= epsilon):
            break
        
        print("2")
        f_term = f_newterm
        g_term = g_newterm
    
    Ai = (alpha*sum_f - beta*sum_g)
    Bi = np.sqrt(3) * (alpha*sum_f + beta*sum_g)
    
    return Ai, Bi

"""
def Maclaurian2(x):
    global alpha, beta, epsilon

    f_term, g_term = 1., x
    sum_f, sum_g = 0., 0.

    while abs(f_term)
"""


Ai_maclaurian = np.zeros_like(x_values)
Bi_maclaurian = np.zeros_like(x_values)
for i, x in enumerate(x_values):
    Ai1, Bi1 = Maclaurian(x, max_terms=10)
    Ai_maclaurian[i] = Ai1    
    Bi_maclaurian[i] = Bi1


# Error calculation ____________________________________________________________
abs_err_ai_mac = np.abs(Ai_maclaurian - scipy_ai_values)
rel_err_ai_mac = np.abs(Ai_maclaurian - scipy_ai_values) / scipy_ai_values

abs_err_bi_mac = np.abs(Ai_maclaurian - scipy_bi_values)
rel_err_bi_mac = np.abs(Ai_maclaurian - scipy_bi_values) / scipy_bi_values


# Plotting of the functions ____________________________________________________

plt.figure(figsize=(15, 8))


plt.subplot(2, 3, 1)
plt.plot(x_values, scipy_ai_values, 'k--', label='Scipy Ai(x)', linewidth=2)
plt.plot(x_values, Ai_maclaurian, label='Maclaurian Ai(x)', linewidth=2)

plt.xlabel('x')
plt.ylabel('Values')
plt.ylim((-0.6, 0.6))
plt.legend()
plt.title('Maclaurian Ai(x)')

plt.subplot(2, 3, 2)
plt.plot(x_values, abs_err_ai_mac, label='Abs. Err. Maclaurian', linewidth=2)
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.yscale('log')
plt.legend()
plt.title('Absolute Error in Ai(x)')

plt.subplot(2, 3, 3)
plt.plot(x_values, rel_err_ai_mac, label='Rel. Err. Maclaurian', linewidth=2)
plt.xlabel('x')
plt.ylabel('Relative Error')
plt.yscale('log')
plt.legend()
plt.title('Relative Error in Ai(x)')


plt.subplot(2, 3, 4)
plt.plot(x_values, scipy_bi_values, 'k--', label='Scipy Bi(x)', linewidth=2)
plt.plot(x_values, Bi_maclaurian, label='Maclaurian Bi(x)', linewidth=2)

plt.xlabel('x')
plt.ylabel('Values')
plt.ylim((-0.6, 0.6))
plt.legend()
plt.title('Maclaurian Bi(x)')

plt.subplot(2, 3, 5)
plt.plot(x_values, abs_err_bi_mac, label='Abs. Err. Maclaurian', linewidth=2)
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.yscale('log')
plt.legend()
plt.title('Absolute Error in Bi(x)')

plt.subplot(2, 3, 6)
plt.plot(x_values, rel_err_bi_mac, label='Rel. Err. Maclaurian', linewidth=2)
plt.xlabel('x')
plt.ylabel('Relative Error')
plt.yscale('log')
plt.legend()
plt.title('Relative Error in Bi(x)')


plt.tight_layout()
plt.show()

