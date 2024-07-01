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

alpha, beta, _, _ = scipy_airy(0)  # faktorja iz Taylorjevega razvoja
beta = -beta

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

def asymptotic_term(s, prev_term):
    # just 1 term of the u_s(x)
    if s == 0:
        return 1
    multiplier = ((3*s - 1/2) * (3*s - 3/2) * (3*s - 5/2)) / (54*s*(s - 1/2))
    return multiplier*prev_term


# Finding the optimal cutoff point _____________________________________________


def Maclaurian(x, cutoff=20, max_terms=500):
    global alpha, beta, epsilon

    sum_f, sum_g, f_term, g_term = 0.0, 0.0, 0.0, 0.0
    for n in range(max_terms):
        f_newterm = maclaurian_term_f(x, n, f_term)
        g_newterm = maclaurian_term_g(x, n, g_term)
        sum_f += f_newterm
        sum_g += g_newterm

        if (abs(f_newterm - f_term) <= epsilon) or (abs(g_newterm - g_term) <= epsilon):
            break
        else:
            f_term = f_newterm
            g_term = g_newterm
    
    Ai = (alpha*sum_f - beta*sum_g)
    Bi = np.sqrt(3) * (alpha*sum_f + beta*sum_g)

    return Ai, Bi

def L(z, max_terms=500):
    sum_L, L_term = 0.0, 0.0
    for n in range(max_terms):
        L_newterm = asymptotic_term(n, L_term)
        sum_L += L_newterm/z

        if abs(L_newterm - L_term) <= epsilon:
            return sum_L
    
    return sum_L

def PandQ(z, max_terms=500):
    uP_term, uQ_term = 0.0, 0.0
    P_term, Q_term = float("inf"), float("inf")
    sum_P, sum_Q = 0.0, 0.0

    for n in range(max_terms):
        uP_term = asymptotic_term(2*n, uP_term)
        uQ_term = asymptotic_term(2*n+1, uQ_term)
        P_newterm = ((-1)**n)*uP_term/(z**(2*n))
        Q_newterm = ((-1)**n)*uP_term/(z**(2*n+1))
        sum_P += P_newterm
        sum_Q += Q_newterm

        if (abs(P_newterm - P_term) <= epsilon) or (abs(Q_newterm - Q_term) <= epsilon):
            return sum_P, sum_Q

        P_term, Q_term = P_newterm, Q_newterm

    return sum_P, sum_Q

def PandQ2(z, max_terms=500):
    sum_P, sum_Q, P_term, Q_term = 0.0, 0.0, 0.0, 0.0

    for n in range(max_terms):
        P_newterm = asymptotic_term(2*n, P_term)
        Q_newterm = asymptotic_term(2*n+1, Q_term)
        sum_P += -P_newterm/(z**2)
        sum_Q += -Q_newterm/(z**2)

        if (abs(P_newterm - P_term) <= epsilon) or (abs(Q_newterm - Q_term) <= epsilon):
            return sum_P, sum_Q
        
        P_term = P_newterm
        Q_term = Q_newterm

    return sum_P, sum_Q


def Asymptotic(x, cutoff=10.0, max_terms=50):
    xi = 2/3 * abs(x)**(3/2)
    
    if x > cutoff:
        a = (np.sqrt(np.pi) * (x)**(1/4))
        Ai = np.exp(-xi)/(2*a) * L(-xi, max_terms)
        Bi = np.exp(xi)/a * L(xi, max_terms)
        return Ai, Bi    
    
    elif x <= -cutoff:
        b = (np.sqrt(np.pi) * (-x)**(1/4))

        #P, Q = PandQ(xi, max_terms)
        P, Q = PandQ(xi, max_terms)
        Ai = (np.sin(xi - np.pi/4)*Q + np.cos(xi - np.pi/4)*P) / b
        Bi = (-np.sin(xi - np.pi/4)*P + np.cos(xi - np.pi/4)*Q) / b
        return Ai, Bi

    return None, None
    

# ______________________________________________________________________________
max_terms = 50 # implemented for safety reasons
cutoff = 3

Ai_maclaurian = np.zeros_like(x_values)
Bi_maclaurian = np.zeros_like(x_values)

Ai_asymptotic = np.zeros_like(x_values)
Bi_asymptotic = np.zeros_like(x_values)
for i, x in enumerate(x_values):
    Ai1, Bi1 = Maclaurian(x, max_terms=max_terms)
    Ai2, Bi2 = Asymptotic(x, cutoff, max_terms=30)
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

