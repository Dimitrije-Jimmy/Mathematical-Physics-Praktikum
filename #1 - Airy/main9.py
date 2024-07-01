import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Importing of the reference Scipy Airy functions ______________________________

from scipy.special import airy as scipy_airy

# Define the range of x values
epsilon = 1e-11
start, end, amount = -15.0, 10.0, 400
x_values = np.linspace(start, end, amount)
#x_pos = np.linspace(start, 0, amount/2)
#x_neg = np.linspace(epsilon, end, amount/2)

# Initialize arrays to store the results
scipy_ai_values, scipy_bi_values = np.zeros_like(x_values), np.zeros_like(x_values)

# Calculate Airy functions using scipy.special.airy and mpmath.airy
for i, x in enumerate(x_values):
    scipy_ai, scipy_aip, scipy_bi, scipy_bip = scipy_airy(x)
    scipy_ai_values[i], scipy_bi_values[i] = scipy_ai, scipy_bi


# Implementing the Maclaurian and Asimptotic expansion _________________________

#alpha, beta = dc.Decimal(0.355028053887817239), dc.Decimal(0.258819403792806798)
alpha = 0.355028053887817239  # alpha = Ai(0) = Bi(0)/3^(1/2)
beta = 0.258819403792806798  # beta = Bi'(0)/3^(1/2) = -Ai'(0)

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

def asymptotic_term(s, prev_term):
    # just 1 term of the u_s(x)
    if s == 0:
        return 1
    multiplier = ((3*s - 1/2) * (3*s - 3/2) * (3*s - 5/2)) / (54*s*(s - 1/2))
    return multiplier*prev_term

def asymptotic_term2(s):
    # just 1 term of the u_s(x)
    if s == 0:
        return 1
    else:
        multiplier = ((3*s - 1/2) * (3*s - 3/2) * (3*s - 5/2)) / (54*s*(s - 1/2))
        return multiplier*asymptotic_term2(s-1)


def L_func(z, max_terms):
    sum_L = 0.
    L_term = 1.
    L_old = float("inf")

    n = 1
    while (abs(L_old - L_term) >= epsilon) and (n <= max_terms):
        L_old = L_term
        L_term = asymptotic_term2(n)/z
        sum_L += L_term
        n += 1
    return sum_L

def L_func2(z, max_terms):
    sum_L = 0.
    for n in range(max_terms):
        L_term = asymptotic_term2(n)/(z**n)
        sum_L += L_term
    return sum_L


def P_func(z, max_terms):
    sum_P = 0.
    P_term = 1.
    P_old = float("inf")

    n = 1
    while (abs(P_old - P_term) >= epsilon) and (n <= max_terms):
        P_old = P_term
        P_term = (-1)*asymptotic_term2(2*n)/(z**2)
        sum_P += P_term
        n += 1
    return sum_P

def P_func2(z, max_terms):
    sum_P = 0.
    for n in range(max_terms):
        P_term = ((-1)**n)*asymptotic_term2(2*n)/(z**(2*n))
        sum_P += P_term
    return sum_P


def Q_func(z, max_terms):
    sum_Q = 0.
    Q_term = 1.
    Q_old = float("inf")

    n = 1
    while (abs(Q_old - Q_term) >= epsilon) and (n <= max_terms):
        Q_old = Q_term
        Q_term = (-1)*asymptotic_term2(2*n+1)/(z**2)
        sum_Q += Q_term
        n += 1
    return sum_Q

def Q_func2(z, max_terms):
    sum_Q = 0.
    for n in range(max_terms):
        Q_term = ((-1)**n)*asymptotic_term2(2*n+1)/(z**(2*n+1))
        sum_Q += Q_term
    return sum_Q


def Asymptotic(x, cutoff, max_terms):
    xi = 2/3 * (abs(x)**(3/2))
    
    if x > cutoff:
        a = (np.sqrt(np.pi) * ((x)**(1/4)))
        #Ai = np.exp(-xi)/(2*a) * L_func(-xi, max_terms)
        Ai = np.exp(-xi)/(2*a) * L_func2(-xi, max_terms)
        #Bi = np.exp(xi)/a * L_func(xi, max_terms)
        Bi = np.exp(xi)/a * L_func2(xi, max_terms)
        return Ai, Bi    
    
    elif x <= -cutoff:
        b = (np.sqrt(np.pi) * ((-x)**(1/4)))
        #P, Q = P_func(xi, max_terms), Q_func(xi, max_terms)
        P, Q = P_func2(xi, max_terms), Q_func2(xi, max_terms)
        Ai = (np.sin(xi - np.pi/4)*Q + np.cos(xi - np.pi/4)*P) / b
        Bi = (-np.sin(xi - np.pi/4)*P + np.cos(xi - np.pi/4)*Q) / b
        return Ai, Bi

    else:
        return None, None
    
"""
def Asymptotic2(x, cutoff, max_terms):
    xi = 2/3 * abs(x)**(3/2)

    if x > cutoff:
        a = ((np.pi**0.5) * (x)**(1/4))   
    
        L1_term, L2_term = 1., 1.
        Ai, Bi = 0., 0.
        Ai_old, Bi_old = float("inf"), float("inf")

        n = 1
        while (abs(Ai_old - Ai) >= epsilon or abs(Bi_old - Bi) >= epsilon) and (n <= max_terms):
            L1_term = asymptotic_term(n)/xi
            L2_term = asymptotic_term(n)/(-xi)

            Ai += np.exp(-xi)/(2*a) * L1_term
            Bi += np.exp(-xi)/a * L2_term
            n += 1

        Ai, Bi = 0., 0.
        Ai_old, Bi_old = float("inf"), float("inf")


    elif x <= cutoff:
        b = ((np.pi**0.5) * (-x)**(1/4))

        P_term, Q_term = 1., 1.
        Ai, Bi = 0., 0.
        Ai_old, Bi_old = float("inf"), float("inf")

        n = 1
        while (abs(Ai_old - Ai) >= epsilon or abs(Bi_old - Bi) >= epsilon) and (n <= max_terms):
            

            Ai = (np.sin(xi - np.pi/4)*Q + np.cos(xi - np.pi/4)*P) / b
            Bi = (-np.sin(xi - np.pi/4)*P + np.cos(xi - np.pi/4)*Q) / b
            n += 1

    return Ai, Bi
"""


# ______________________________________________________________________________
max_terms = 50 # implemented for safety reasons
cutoff = 3

Ai_maclaurian = np.zeros_like(x_values)
Bi_maclaurian = np.zeros_like(x_values)

Ai_asymptotic = np.zeros_like(x_values)
Bi_asymptotic = np.zeros_like(x_values)
for i, x in enumerate(x_values):
    Ai1, Bi1, _ = Maclaurian(x, max_terms=50)
    Ai2, Bi2 = Asymptotic(x, cutoff, max_terms=20)
    Ai_maclaurian[i] = Ai1
    Bi_maclaurian[i] = Bi1
    Ai_asymptotic[i] = Ai2        
    Bi_asymptotic[i] = Bi2


# Error calculation ____________________________________________________________
abs_err_ai_mac = np.abs(Ai_maclaurian - scipy_ai_values)
rel_err_ai_mac = np.abs(Ai_maclaurian - scipy_ai_values) / scipy_ai_values

abs_err_ai_asy = np.abs(Ai_asymptotic - scipy_ai_values)
rel_err_ai_asy = np.abs(Ai_asymptotic - scipy_ai_values) / scipy_ai_values


abs_err_bi_mac = np.abs(Bi_maclaurian - scipy_bi_values)
rel_err_bi_mac = np.abs(Bi_maclaurian - scipy_bi_values) / scipy_bi_values

abs_err_bi_asy = np.abs(Bi_asymptotic - scipy_bi_values)
rel_err_bi_asy = np.abs(Bi_asymptotic - scipy_bi_values) / scipy_bi_values



# Plotting of the functions ____________________________________________________

plt.figure(figsize=(15, 8))


plt.subplot(2, 3, 1)
plt.axhline(y=0.0, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=0.0, color='k', linestyle='--', alpha=0.5)
plt.plot(x_values, scipy_ai_values, 'k-', alpha=0.7, label='Scipy Ai(x)', linewidth=2)
#plt.plot(x_values, Ai_maclaurian, label='Maclaurian Ai(x)', linewidth=2)
#plt.scatter(x_values, Ai_maclaurian, color='red', s=10, label='Maclaurian Ai(x)')
plt.scatter(x_values, Ai_asymptotic, color='orange', s=5, label="Asymptotic Ai(x)")

#plt.grid()
plt.xlabel('x')
plt.ylabel('Values')
plt.ylim((-0.6, 0.6))
plt.legend()
plt.title('Maclaurian Ai(x)')

plt.subplot(2, 3, 2)
plt.plot(x_values, abs_err_ai_mac, label='Abs. Err. Maclaurian', linewidth=2)
plt.plot(x_values, abs_err_ai_asy, label='Abs. Err. Maclaurian', linewidth=2)
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.yscale('log')
plt.legend()
plt.title('Absolute Error in Ai(x)')

plt.subplot(2, 3, 3)
plt.plot(x_values, rel_err_ai_mac, label='Rel. Err. Maclaurian', linewidth=2)
plt.plot(x_values, rel_err_ai_asy, label='Rel. Err. Maclaurian', linewidth=2)
plt.xlabel('x')
plt.ylabel('Relative Error')
plt.yscale('log')
plt.legend()
plt.title('Relative Error in Ai(x)')


plt.subplot(2, 3, 4)
plt.axhline(y=0.0, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=0.0, color='k', linestyle='--', alpha=0.5)
plt.plot(x_values, scipy_bi_values, 'k-', alpha=0.7, label='Scipy Bi(x)', linewidth=2)
#plt.plot(x_values, Bi_maclaurian, label='Maclaurian Bi(x)', linewidth=2)
#plt.scatter(x_values, Bi_maclaurian, color='red', s=5, label='Maclaurian Bi(x)')
plt.scatter(x_values, Bi_asymptotic, color='orange', s=5, label="Asymptotic Bi(x)")


plt.xlabel('x')
plt.ylabel('Values')
plt.ylim((-0.6, 0.6))
plt.legend()
plt.title('Maclaurian Bi(x)')

plt.subplot(2, 3, 5)
plt.plot(x_values, abs_err_bi_mac, label='Abs. Err. Maclaurian', linewidth=2)
plt.plot(x_values, abs_err_bi_asy, label='Abs. Err. Maclaurian', linewidth=2)
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.yscale('log')
plt.legend()
plt.title('Absolute Error in Bi(x)')

plt.subplot(2, 3, 6)
plt.plot(x_values, rel_err_bi_mac, label='Rel. Err. Maclaurian', linewidth=2)
plt.plot(x_values, rel_err_bi_asy, label='Rel. Err. Maclaurian', linewidth=2)
plt.xlabel('x')
plt.ylabel('Relative Error')
plt.yscale('log')
plt.legend()
plt.title('Relative Error in Bi(x)')


plt.tight_layout()
plt.show()



# Plotting individual figures __________________________________________________

fig = plt.figure(figsize=(5, 4.5))

"""
# s-ti člen vsote za z=12
max_terms_iteration = np.arange(0, 40)
L_test = []
P_test = []
Q_test = []
for i, iteration in enumerate(max_terms_iteration):
    L_test.append(L_func2(12, iteration))
    P_test.append(P_func2(12, iteration))
    Q_test.append(Q_func2(12, iteration))


plt.figure(figsize=(15, 8))

plt.scatter(max_terms_iteration, L_test, color='green', s=5, label="L")
plt.scatter(max_terms_iteration, P_test, color='orange', s=5, label="P")
plt.scatter(max_terms_iteration, Q_test, color='red', s=5, label="Q")

#plt.grid()
plt.xlabel('s')
plt.ylabel('s-ti člen vsote za z=12')
#plt.ylim((-0.6, 0.6))
plt.legend()
plt.title('Test')

plt.tight_layout()
plt.show()
"""



# Relative Error _______________________________________________________________
plt.figure(figsize=(13, 5))

plt.subplot(1, 2, 1)


plt.plot(t, rel_err_ai_frm, 'm-', alpha=0.5)
plt.scatter(t, rel_err_ai_frm, s=5, label='Numerično')
plt.plot(t, rel_err_ai_asy, 'g-', alpha=0.5)
plt.scatter(t, rel_err_ai_asy, s=5, label='Asimptotsko')
plt.xlabel('x')
plt.ylabel(r'$\rho = \vert Ai_{num} - Ai_{ref} \vert \;/\; Ai_{ref}$')
#plt.yscale('log')
#plt.ylim((1e-17, 1e5))
plt.legend()
plt.title('Relativna napaka Ai(x)')


plt.subplot(1, 2, 2)


plt.plot(t, rel_err_bi_frm, 'm-', alpha=0.5)
plt.scatter(t, rel_err_bi_frm, s=5, label='Numerično')
plt.plot(t, rel_err_bi_asy, 'g-', alpha=0.5)
plt.scatter(t, rel_err_bi_asy, s=5, label='Asimptotsko')
plt.xlabel('x')
plt.ylabel(r'$\rho = \vert Bi_{num} - Bi_{ref} \vert \;/\; Bi_{ref}$')
#plt.yscale('log')
#plt.ylim((1e-17, 1e5))
plt.legend()
plt.title('Relativna napaka Bi(x)')

plt.tight_layout()
plt.show()
plt.clf()
plt.close()
