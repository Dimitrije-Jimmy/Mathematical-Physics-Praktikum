import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Importing of the reference Scipy Airy functions ______________________________

from scipy.special import airy as scipy_airy
from scipy.special import ai_zeros
from scipy.special import bi_zeros


nt = 100  # number of zeroes to compute

scipy_ai_zeros, _, _, _ = ai_zeros(nt)
scipy_bi_zeros, _, _, _ = bi_zeros(nt)


print(scipy_ai_zeros)

from scipy.special import airy as scipy_airy

# Define the range of x values
epsilon = 1e-11
start, end, amount = -62.0, 0, 4000
x_values = np.linspace(start, end, amount)


# Initialize arrays to store the results
scipy_ai_values, scipy_bi_values = np.zeros_like(x_values), np.zeros_like(x_values)

# Calculate Airy functions using scipy.special.airy and mpmath.airy
for i, x in enumerate(x_values):
    scipy_ai, _, scipy_bi, _ = scipy_airy(x)
    scipy_ai_values[i], scipy_bi_values[i] = scipy_ai, scipy_bi


# Implementacija iz navodil ____________________________________________________


def f(z):
    u1 = 5/(48*z**2)
    u2 = 5/(36*z**4)
    u3 = 77125/(82944*z**6)
    u4 = 108056875/(6967296*z**8)
    return (z**(2/3))*(1 + u1 - u2 + u3 - u4)

def nicle(s):
    factor1 = 3*np.pi*(4*s - 1)/8
    factor2 = 3*np.pi*(4*s - 3)/8
    return -f(factor1), -f(factor2)


nicle_ai, nicle_bi = [], []
for n in range(1, nt+1):
    ai, bi = nicle(n)
    nicle_ai.append(ai)
    nicle_bi.append(bi)

#print(nicle_ai)

# Implementing the Maclaurian and Asimptotic expansion _________________________

#alpha = 0.355028053887817239  
#beta = 0.258819403792806798  

alpha, beta, _, _ = scipy_airy(0)
beta = -beta

def Maclaurian(x, max_terms):
    global alpha, beta, epsilon

    f_term, g_term = 1., x

    Ai = alpha*f_term - beta*g_term
    Bi = (3.**0.5)*(alpha*f_term + beta*g_term)

    Ai_old, Bi_old = float("inf"), float("inf")

    n = 1
    while (abs(Ai_old - Ai) >= epsilon or abs(Bi_old - Bi) >= epsilon) and (n <= max_terms):
        Ai_old, Bi_old = Ai, Bi

        f_multiplier = (n-2/3) * (x**3) / (n * (3*n-1) * (3*n - 2))
        g_multiplier = (n-1/3) * (x**3) / ((3*n+1) * n * (3*n-1))

        f_term *= f_multiplier
        g_term *= g_multiplier

        Ai += alpha*f_term - beta*g_term
        Bi += (3.**0.5)*(alpha*f_term + beta*g_term)

        n += 1

    return Ai, Bi, n

def asymptotic_term2(s):
    # the entire u_s(x)
    if s == 0:
        return 1
    else:
        multiplier = ((3*s - 1/2) * (3*s - 3/2) * (3*s - 5/2)) / (54*s*(s - 1/2))
        return multiplier*asymptotic_term2(s-1)


def L_func2(z, max_terms):
    sum_L = 0.
    for n in range(max_terms):
        L_term = asymptotic_term2(n)/(z**n)
        sum_L += L_term
    return sum_L

def P_func2(z, max_terms):
    sum_P = 0.
    for n in range(max_terms):
        P_term = ((-1)**n)*asymptotic_term2(2*n)/(z**(2*n))
        sum_P += P_term
    return sum_P

def Q_func2(z, max_terms):
    sum_Q = 0.
    for n in range(max_terms):
        Q_term = ((-1)**n)*asymptotic_term2(2*n+1)/(z**(2*n+1))
        sum_Q += Q_term
    return sum_Q


def Asymptotic(x, cutoff, max_terms):
    xi = 2/3 * (abs(x)**(3/2))
    
    if x <= cutoff:
        b = (mp.sqrt(mp.pi) * ((-x)**(1/4)))
        P, Q = P_func2(xi, max_terms), Q_func2(xi, max_terms)
        Ai = (mp.sin(xi - mp.pi/4)*Q + mp.cos(xi - mp.pi/4)*P) / b
        Bi = (-mp.sin(xi - mp.pi/4)*P + mp.cos(xi - mp.pi/4)*Q) / b
        return Ai, Bi

    else:
        return None, None


# ______________________________________________________________________________
#max_terms = 50 # implemented for safety reasons
cutoff = -7.074

Ai_values = np.zeros_like(x_values)
Bi_values = np.zeros_like(x_values)
for i, x in enumerate(x_values):
    if x >= cutoff:
        Ai, Bi, _ = Maclaurian(x, max_terms=50)
        Ai_values[i] = Ai
        Bi_values[i] = Bi
    else:
        Ai, Bi = Asymptotic(x, cutoff, max_terms=20)
        Ai_values[i] = Ai
        Bi_values[i] = Bi


def Ai_funkcija(x):
    cutoff = -7.074
    if x >= cutoff:
        value, _, _ = Maclaurian(x, max_terms=50)
        return value
    else:  
        value, _ = Asymptotic(x, cutoff, max_terms=20)
        return value
    
def Bi_funkcija(x):
    cutoff = -7.074
    if x >= cutoff:
        _, value, _ = Maclaurian(x, max_terms=50)
        return value
    else:  
        _, value = Asymptotic(x, cutoff, max_terms=20)
        return value


zeros_frm_ai = []
for i, x_guess in enumerate(nicle_ai):
    zero = mp.findroot(Ai_funkcija, x_guess, tol=1e-10)
    zeros_frm_ai.append(float(mp.nstr(zero)))
zeros_frm_ai = np.array(zeros_frm_ai)

print(zeros_frm_ai)

zeros_frm_bi = []
for i, x_guess in enumerate(nicle_bi):
    zero = mp.findroot(Bi_funkcija, x_guess, tol=1e-10)
    zeros_frm_bi.append(float(mp.nstr(zero)))
zeros_frm_bi = np.array(zeros_frm_bi)

print(len(zeros_frm_bi))



# Errors _______________________________________________________________________

abs_err_ai_asy = np.abs(nicle_ai - scipy_ai_zeros)
#rel_err_ai_asy = abs_err_ai_asy / scipy_ai_zeros

abs_err_ai_frm = np.abs(zeros_frm_ai - scipy_ai_zeros)
#rel_err_ai_frm = abs_err_ai_frm / scipy_ai_zeros


abs_err_bi_asy = np.abs(nicle_bi - scipy_bi_zeros)
#rel_err_bi_asy = abs_err_bi_asy / scipy_bi_zeros

abs_err_bi_frm = np.abs(zeros_frm_bi - scipy_bi_zeros)
#rel_err_bi_frm = abs_err_bi_frm / scipy_bi_zeros


# Plotting the results _________________________________________________________

t = np.arange(1, nt+1)


plt.figure(figsize=(13, 5))


plt.subplot(1, 2, 1)


plt.plot(t, scipy_ai_zeros, 'm-', alpha=0.5, linewidth=0.7, label='Scipy')
plt.scatter(t, zeros_frm_ai, label='Numerično')
plt.scatter(t, nicle_ai, s=5, label='Asimptotsko')
plt.xlabel('t')
plt.ylabel(r'$x_0$')
plt.legend()
plt.title('Ničle Ai(x)')


plt.subplot(1, 2, 2)


plt.plot(t, scipy_bi_zeros, 'm-', alpha=0.5, linewidth=0.7, label='Scipy')
plt.scatter(t, zeros_frm_bi, label='Numerično')
plt.scatter(t, nicle_bi, s=5, label='Asimptotsko')
plt.xlabel('t')
plt.ylabel(r'$x_0$')
plt.legend()
plt.title('Ničle Bi(x)')

plt.tight_layout()
plt.show()
plt.clf()
plt.close()


# Absolute Error _______________________________________________________________
plt.figure(figsize=(13, 5))


plt.subplot(1, 2, 1)

plt.axhline(y=1e-10, color='k', linestyle='--', alpha=0.5)

plt.plot(t, abs_err_ai_frm, 'm-', alpha=0.5)
plt.scatter(t, abs_err_ai_frm, s=5, label='Numerično')
plt.plot(t, abs_err_ai_asy, 'g-', alpha=0.5)
plt.scatter(t, abs_err_ai_asy, s=5, label='Asimptotsko')
plt.xlabel(r'ničle - $\{a_s\}_{s=1}^{100}$')
plt.ylabel(r'$\Delta = \vert Ai_{num} - Ai_{ref} \vert$')
plt.yscale('log')
#plt.ylim((1e-17, 1e5))
plt.legend()
plt.title('Absolutna napaka ničel Ai(x)')


plt.subplot(1, 2, 2)

plt.axhline(y=1e-10, color='k', linestyle='--', alpha=0.5)

plt.plot(t, abs_err_bi_frm, 'm-', alpha=0.5)
plt.scatter(t, abs_err_bi_frm, s=5, label='Numerična')
plt.plot(t, abs_err_bi_asy, 'g-', alpha=0.5)
plt.scatter(t, abs_err_bi_asy, s=5, label='Asimptotsko')
plt.xlabel(r'ničle - $\{b_s\}_{s=1}^{100}$')
plt.ylabel(r'$\Delta = \vert Bi_{num} - Bi_{ref} \vert$')
plt.yscale('log')
#plt.ylim((1e-17, 1e5))
plt.legend()
plt.title('Absolutna napaka ničel Bi(x)')

plt.tight_layout()
plt.show()
plt.clf()
plt.close()

