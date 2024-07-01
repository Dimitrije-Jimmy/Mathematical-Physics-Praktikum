import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# Importing of the reference Scipy Airy functions ______________________________

from scipy.special import airy as scipy_airy

# Define the range of x values
epsilon = 1e-11
start, end, amount = -15.0, 10.0, 400
x_values = np.linspace(start, end, amount)


# Initialize arrays to store the results
scipy_ai_values, scipy_bi_values = np.zeros_like(x_values), np.zeros_like(x_values)

# Calculate Airy functions using scipy.special.airy and mpmath.airy
for i, x in enumerate(x_values):
    scipy_ai, scipy_aip, scipy_bi, scipy_bip = scipy_airy(x)
    scipy_ai_values[i], scipy_bi_values[i] = scipy_ai, scipy_bi


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

def asymptotic_term(s, prev_term):
    # just 1 term of the u_s(x)
    if s == 0:
        return 1
    multiplier = ((3*s - 1/2) * (3*s - 3/2) * (3*s - 5/2)) / (54*s*(s - 1/2))
    return multiplier*prev_term

def asymptotic_term2(s):
    # the entire u_s(x)
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
        Ai = np.exp(-xi)/(2*a) * L_func2(-xi, max_terms)
        Bi = np.exp(xi)/a * L_func2(xi, max_terms)
        return Ai, Bi    
    
    elif x <= -cutoff:
        b = (np.sqrt(np.pi) * ((-x)**(1/4)))
        P, Q = P_func2(xi, max_terms), Q_func2(xi, max_terms)
        Ai = (np.sin(xi - np.pi/4)*Q + np.cos(xi - np.pi/4)*P) / b
        Bi = (-np.sin(xi - np.pi/4)*P + np.cos(xi - np.pi/4)*Q) / b
        return Ai, Bi

    else:
        return None, None
    

def Asymptotic2(x, cutoff1, cutoff2, max_terms):
    xi = 2/3 * (abs(x)**(3/2))
    
    if x > cutoff2:
        a = (np.sqrt(np.pi) * ((x)**(1/4)))
        Ai = np.exp(-xi)/(2*a) * L_func2(-xi, max_terms)
        Bi = np.exp(xi)/a * L_func2(xi, max_terms)
        return Ai, Bi    
    
    elif x <= cutoff1:
        b = (np.sqrt(np.pi) * ((-x)**(1/4)))
        P, Q = P_func2(xi, max_terms), Q_func2(xi, max_terms)
        Ai = (np.sin(xi - np.pi/4)*Q + np.cos(xi - np.pi/4)*P) / b
        Bi = (-np.sin(xi - np.pi/4)*P + np.cos(xi - np.pi/4)*Q) / b
        return Ai, Bi

    else:
        return None, None


# ______________________________________________________________________________
#max_terms = 50 # implemented for safety reasons
cutoff = 1

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
rel_err_ai_mac = abs_err_ai_mac / scipy_ai_values

abs_err_ai_asy = np.abs(Ai_asymptotic - scipy_ai_values)
rel_err_ai_asy = abs_err_ai_asy / scipy_ai_values


abs_err_bi_mac = np.abs(Bi_maclaurian - scipy_bi_values)
rel_err_bi_mac = abs_err_bi_mac / scipy_bi_values

abs_err_bi_asy = np.abs(Bi_asymptotic - scipy_bi_values)
rel_err_bi_asy = abs_err_bi_asy / scipy_bi_values


# Plotting of the functions ____________________________________________________


# Optimal cutoff point _________________________________________________________
plt.figure(figsize=(8, 5))

plt.axhline(y=1e-10, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=-7.074, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=5.5478, color='k', linestyle='--', alpha=0.5)

plt.plot(x_values, abs_err_ai_mac, 'm-', alpha=0.5)
plt.scatter(x_values, abs_err_ai_mac, s=5, label='Maclaurian')
plt.plot(x_values, abs_err_ai_asy, 'g-', alpha=0.5)
plt.scatter(x_values, abs_err_ai_asy, s=5, label='Asimptotsko')
plt.xlabel('x')
plt.ylabel(r'$|Ai_{num} - Ai_{ref}|$')
plt.yscale('log')
plt.ylim((1e-17, 1e0))
plt.legend()
plt.title('Absolutna napaka Ai(x)')

plt.tight_layout()
plt.show()
plt.clf()
plt.close()

plt.figure(figsize=(8, 5))

plt.axhline(y=1e-10, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=-7.074, color='k', linestyle='--', alpha=0.5)
#plt.axvline(x=5.5478, color='k', linestyle='--', alpha=0.5)

plt.plot(x_values, abs_err_bi_mac, 'm-', alpha=0.5)
plt.scatter(x_values, abs_err_bi_mac, s=5, label='Maclaurian')
plt.plot(x_values, abs_err_bi_asy, 'g-', alpha=0.5)
plt.scatter(x_values, abs_err_bi_asy, s=5, label='Asimptotsko')
plt.xlabel('x')
plt.ylabel(r'$|Bi_{num} - Bi_{ref}|$')
plt.yscale('log')
plt.ylim((1e-17, 1e0))
plt.legend()
plt.title('Absolutna napaka Bi(x)')

plt.tight_layout()
plt.show()
plt.clf()
plt.close()


# Relative Error _______________________________________________________________
plt.figure(figsize=(13, 5))

plt.subplot(1, 2, 1)

plt.axhline(y=1e-10, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=-7.074, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=5.5478, color='k', linestyle='--', alpha=0.5)

plt.plot(x_values, rel_err_ai_mac, 'm-', alpha=0.5)
plt.scatter(x_values, rel_err_ai_mac, s=5, label='Maclaurian')
plt.plot(x_values, rel_err_ai_asy, 'g-', alpha=0.5)
plt.scatter(x_values, rel_err_ai_asy, s=5, label='Asimptotsko')
plt.xlabel('x')
plt.ylabel(r'$\vert Ai_{num} - Ai_{ref} \vert \;/\; Ai_{ref}$')
plt.yscale('log')
plt.ylim((1e-17, 1e5))
plt.legend()
plt.title('Relativna napaka Ai(x)')


plt.subplot(1, 2, 2)

plt.axhline(y=1e-10, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=-7.074, color='k', linestyle='--', alpha=0.5)
#plt.axvline(x=5.5478, color='k', linestyle='--', alpha=0.5)

#plt.plot(x_values, rel_err_bi_mac, 'm-', alpha=0.5)
plt.scatter(x_values, rel_err_bi_mac, s=5, label='Maclaurian')
#plt.plot(x_values, rel_err_bi_asy, 'g-', alpha=0.5)
plt.scatter(x_values, rel_err_bi_asy, s=5, label='Asimptotsko')
plt.xlabel('x')
plt.ylabel(r'$\vert Bi_{num} - Bi_{ref} \vert \;/\; Bi_{ref}$')
plt.yscale('log')
plt.ylim((1e-17, 1e5))
plt.legend()
plt.title('Relativna napaka Bi(x)')

plt.tight_layout()
plt.show()
plt.clf()
plt.close()

# Plotting optimal functions ___________________________________________________

cutoff1 = -7.074
cutoff2 = 5.5478


Ai_maclaurian = np.zeros_like(x_values)
Bi_maclaurian = np.zeros_like(x_values)

Ai_asymptotic = np.zeros_like(x_values)
Bi_asymptotic = np.zeros_like(x_values)
for i, x in enumerate(x_values):
    Ai1, Bi1, _ = Maclaurian(x, max_terms=50)
    Ai2, Bi2 = Asymptotic2(x, cutoff1, cutoff2, max_terms=20)
    Ai_maclaurian[i] = Ai1
    Bi_maclaurian[i] = Bi1
    Ai_asymptotic[i] = Ai2        
    Bi_asymptotic[i] = Bi2


index_Ai = np.where(np.isnan(Ai_asymptotic))[0]
index1, index2 = index_Ai[0], index_Ai[-1]

index_Bi = np.where(np.isnan(Bi_asymptotic))[0][0]

print(index1, index2, index_Bi)


plt.figure(figsize=(12, 6))


plt.subplot(2, 1, 1)
plt.axhline(y=0.0, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=0.0, color='k', linestyle='--', alpha=0.5)
plt.plot(x_values, scipy_ai_values, 'k-', alpha=0.7, label='Scipy Ai(x)', linewidth=1)

plt.scatter(x_values[index1:index2+1], Ai_maclaurian[index1:index2+1], color='blue', s=5, label="Maclaurian")
plt.scatter(x_values[:index1], Ai_asymptotic[:index1], color='cyan', s=5, label="Asimptotsko za x < 0")
plt.scatter(x_values[index2+2:], Ai_asymptotic[index2+2:], color='green', s=5, label="Asimptotsko za x > 0")

#plt.grid()
plt.xlabel('x')
plt.ylabel('Ai(x)')
plt.ylim((-0.6, 0.8))
plt.legend(loc='upper right')
plt.title('Ai(x)')


plt.subplot(2, 1, 2)
plt.axhline(y=0.0, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=0.0, color='k', linestyle='--', alpha=0.5)
plt.plot(x_values, scipy_bi_values, 'k-', alpha=0.7, label='Scipy Bi(x)', linewidth=1)

plt.scatter(x_values[index_Bi:], Bi_maclaurian[index_Bi:], color='orange', s=5, label="Maclaurian")
plt.scatter(x_values[:index_Bi], Bi_asymptotic[:index_Bi], color='red', s=5, label="Asimptotsko")

#plt.grid()
plt.xlabel('x')
plt.ylabel('Bi(x)')
plt.ylim((-0.6, 1))
plt.legend()
plt.title('Bi(x)')

plt.tight_layout()
plt.show()
plt.clf()
plt.close()



