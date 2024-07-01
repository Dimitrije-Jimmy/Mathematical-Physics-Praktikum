import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp


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

alpha, beta = 0.355028053887817239, 0.258819403792806798
#z1, z2 = 1/3, 2/3         # spremeni v Decimal(1)/Decimal(3) da bi blo natancno


def maclaurian_term_f(x, n, prev_term):
    # just 1 term of the f_n(x)
    if n == 0:
        return 1/3
    multiplier = (n-2/3) * (x**3) / (n * (3*n-1) * (3*n - 2))
    return multiplier*prev_term

def maclaurian_term_g(x, n, prev_term):
    # just 1 term of the f_n(x)
    if n == 0:
        return 2/3         # spremeni v Decimal(1)/Decimal(3) da bi blo natancno
    multiplier = (n-1/3) * (x**3) / ((3*n+1) * n * (3*n-1))
    return multiplier*prev_term


def asymptotic_term(x, s, prev_term, is_positive_x):
    # just 1 term of the u_s(x)
    if s == 0:
        return 1
    multiplier = ((3*s - 1/2) * (3*s - 3/2) * (3*s - 5/2)) / (54*s*(s - 1/2))
    return multiplier*prev_term


# Finding the optimal cutoff point _____________________________________________

def Maclaurian_recursion(x, cutoff=20.0, max_terms=50):
    global alpha, beta

    if abs(x) < cutoff:
        sum_f, sum_g, f_term, g_term = 0.0, 0.0, 1.0, 1.0
        for n in range(max_terms):
            f_term = maclaurian_term_f(x, n, f_term)
            g_term = maclaurian_term_g(x, n, g_term)
            sum_f += f_term
            sum_g += g_term
        
        Ai = alpha*sum_f - beta*sum_g
        Bi = mp.sqrt(3) * (alpha*sum_f + beta*sum_g)
        return Ai, Bi
    
    return 0, 0


"""
# Vectorizing function and applying on all elements in array
vfunc = np.vectorize(Airy_Ai_maclaurian)
vfunc(x_values)

https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
Generally avoid np.vectorize, as it does not perform well, and has (or had) a number of issues. 
If you are handling other data types, you may want to investigate the other methods shown below.

import numpy as np
x = np.array([1, 2, 3, 4, 5])
f = lambda x: x ** 2
squares = f(x)
"""
cutoff = 10
max_terms = 500
#f = lambda x: Maclaurian_recursion(x, cutoff, max_terms)
#Ai_maclaurian, Bi_maclaurian = f(x_values)
#mauclarian = [Maclaurian_recursion(x, cutoff, max_terms) for x in x_values]
#print(mauclarian)
#Ai_maclaurian, Bi_maclaurian = mauclarian[:, 0], mauclarian[:, 1]

#Ai_maclaurian = [Maclaurian_recursion(x, cutoff, max_terms) for x in x_values]
#print(Ai_maclaurian)
Ai_maclaurian = []
Bi_maclaurian = []
"""
for x in x_values:
    mauclarian = Maclaurian_recursion(x, cutoff, max_terms)
    if mauclarian == None:
        Ai_maclaurian.append(0)        
        Bi_maclaurian.append(0)
    else:
        Ai, Bi = mauclarian
        Ai_maclaurian.append(Ai)        
        Bi_maclaurian.append(Bi)
"""

def Mac_new_way(x, cutoff=5.0, max_terms=50):

    def recursion(x, n, prev_term):
        if n == 0:
            return 1
        else:
            multiplier = 


Ai_new_way = []
for x in x_values:
    Ai, Bi = Maclaurian_recursion(x, cutoff, max_terms)
    Ai_maclaurian.append(Ai)        
    Bi_maclaurian.append(Bi)
    Ai_new_way.append(Mac_new_way(x, cutoff, max_terms))






# Zlepljena Airy Ai ____________________________________________________________
#def Airy_Ai_recursive(x, cutoff=5.0, max_terms=50):
#    if abs(x) < cutoff:
        

# Calculation of the absolute and relative error _______________________________


# Plotting of the functions ____________________________________________________

plt.figure(figsize=(10, 5))


plt.subplot(1, 2, 1)
plt.plot(x_values, Ai_maclaurian, label='Ai(x)', linewidth=2)
plt.plot(x_values, scipy_ai_values, label='Scipy Ai(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('Values')
plt.ylim((-0.6, 0.6))
plt.legend()
plt.title('Maclaurian Ai(x)')


plt.subplot(1, 2, 2)
plt.plot(x_values, Bi_maclaurian, label='Bi(x)', linewidth=2)
plt.plot(x_values, scipy_bi_values, label='Scipy Bi(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('Values')
plt.ylim((-0.6, 0.6))
plt.legend()
plt.title('Maclaurian Bi(x)')

plt.tight_layout()
plt.show()
