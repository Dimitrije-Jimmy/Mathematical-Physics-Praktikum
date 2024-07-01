import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
import decimal as dc


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
#alpha, beta = 0.355028053887817239, 0.258819403792806798
#z1, z2 = 1/3, 2/3         # spremeni v Decimal(1)/Decimal(3) da bi blo natancno

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


def asymptotic_term(s, prev_term):
    # just 1 term of the u_s(x)
    if s == 0:
        return 1
    multiplier = ((3*s - 1/2) * (3*s - 3/2) * (3*s - 5/2)) / (54*s*(s - 1/2))
    return multiplier*prev_term

# Finding the optimal cutoff point _____________________________________________

def Maclaurian(x, cutoff=10.0, max_terms=50):
    global alpha, beta

    if abs(x) < cutoff:
        sum_f, sum_g, f_term, g_term = 0.0, 0.0, 0.0, 0.0
        for n in range(max_terms):
            f_term = maclaurian_term_f(x, n, f_term)
            g_term = maclaurian_term_g(x, n, g_term)
            sum_f += f_term
            sum_g += g_term
        
        Ai = (alpha*sum_f - beta*sum_g)
        Bi = np.sqrt(3) * (alpha*sum_f + beta*sum_g)
        return Ai, Bi
    
    return 0.0, 0.0

def L(z, max_terms=50):
    sum_L, L_term = 0.0, 0.0
    for n in range(max_terms):
        L_term = asymptotic_term(n, L_term)
        sum_L += L_term/z
    
    return sum_L

def PandQ(z, max_terms=50):
    sum_P, sum_Q, P_term, Q_term = 0.0, 0.0, 0.0, 0.0

    for n in range(max_terms):
        P_term = asymptotic_term(2*n, P_term)
        Q_term = asymptotic_term(2*n+1, Q_term)
        sum_P += -P_term/(z**2)
        sum_Q += -Q_term/(z**2)

    return sum_P, sum_Q


#sez = np.arange(0, 60)
#plt.plot(sez, [L(12, max_terms=s) for s in sez])



def Asymptotic(x, cutoff=10.0, max_terms=50):
    xi = 2/3 * abs(x)**(3/2)
    
    if x >= cutoff:
        a = (np.sqrt(np.pi) * (x)**(1/4))
        Ai = np.exp(-xi)/(2*a) * L(-xi, max_terms)
        Bi = np.exp(-xi)/a * L(xi, max_terms)
        return Ai, Bi    
    
    elif x <= -cutoff:
        b = (np.sqrt(np.pi) * (-x)**(1/4))

        P, Q = PandQ(xi, max_terms)
        Ai = (np.sin(xi - np.pi/4)*Q + np.cos(xi - np.pi/4)*P) / b
        Bi = (-np.sin(xi - np.pi/4)*P + np.cos(xi - np.pi/4)*Q) / b
        return Ai, Bi
    
    return 0, 0


def AsimptotskaVrsta(x):
    ksi = (2. / 3.) * (np.abs(x) ** (3. / 2.))
    j = 1.
    Ai_old = float("inf")
    Bi_old = float("inf")

    if (x >= 0.):
        Lclen = 1.
        faktor1 = np.exp(-ksi) / (2. * np.pi ** 0.5 * x ** 0.25)
        faktor2 = np.exp(ksi) / (np.pi ** 0.5 * x ** 0.25)
        Ai = faktor1
        Bi = faktor2

        while np.abs(Ai - Ai_old) >= epsilon or np.abs(Bi - Bi_old) >= epsilon:
            Ai_old = Ai
            Bi_old = Bi
            # asimptotski razvoj za velike x>0
            # residuumska zveza: Gamma(z+n)/Gamma(z) = z*(z+1)*...*(z+n-1)
            Lclen = Lclen * (3. * (j - 1.) + 5. / 2.) * \
                (3. * (j - 1.) + 3. / 2.) * \
                (3. * (j - 1.) + 1. / 2.) / (54 * j * (j - 0.5) * ksi)
            
            Ai = Ai + faktor1 * (-1) ** j * Lclen
            Bi = Bi + faktor2 * Lclen
            j = j + 1.

    # asimptotski razvoj za velike |x|
    if (x < 0):
        Pclen = 1.
        Qclen = (2. + 0.5) * (1. + 0.5) / (54. * ksi)
        faktor = 1. / (np.pi ** 0.5 * (-x) ** 0.25)
        Ai = faktor * (np.sin(ksi - np.pi / 4.) * Qclen + np.cos(ksi - np.pi / 4.) * Pclen)
        Bi = faktor * (-np.sin(ksi - np.pi / 4.) * Pclen + np.cos(ksi - np.pi / 4.) * Qclen)

        while np.abs(Ai - Ai_old) >= epsilon or np.abs(Bi - Bi_old) >= epsilon:
            Ai_old = Ai
            Bi_old = Bi
            # asimptotski razvoj za velike |x|
            # Gamma(s + 1/2) = (2n)!*(pi^0.5)/((4^n)*n!)
            Pclen = Pclen * (-1.) * (6. * (j - 1.) + 11. / 2.) * (6. * (j - 1.) + 9. / 2.) * \
                (6. * (j - 1.) + 7. / 2.) * (6. * (j - 1.) + 5. / 2.) * \
                (6. * (j - 1.) + 3. / 2.) * (6. * (j - 1.) + 1. / 2.) / \
                (54. ** 2. * 2. * j * (2. * j - 1.) * (2. * (j - 1.) + 3. / 2.) *
                 (2. * (j - 1.) + 1. / 2.) * ksi ** 2.)
            Qclen = Qclen * (-1.) * (6. * (j - 1.) + 17. / 2.) * (6. * (j - 1.) + 15. / 2.) * \
                (6. * (j - 1.) + 13. / 2.) * (6. * (j - 1.) + 11. / 2.) * \
                (6. * (j - 1.) + 9. / 2.) * (6. * (j - 1.) + 7. / 2.) / \
                (54. ** 2. * (2. * j + 1.) * 2. * j * (2. * (j - 1.) + 5. / 2.) *
                 (2. * (j - 1.) + 3. / 2.) * ksi ** 2.)

            Ai = Ai + faktor * (np.sin(ksi - np.pi / 4.) * Qclen + np.cos(ksi - np.pi / 4.) * Pclen)
            Bi = Bi + faktor * (-np.sin(ksi - np.pi / 4.) * Pclen + np.cos(ksi - np.pi / 4.) * Qclen)
            j = j + 1.

    return Ai, Bi


cutoff = 20
max_terms = 500


Ai_maclaurian = np.zeros_like(x_values)
Bi_maclaurian = np.zeros_like(x_values)
Ai_asymptotic = np.zeros_like(x_values)
Bi_asymptotic = np.zeros_like(x_values)
for i, x in enumerate(x_values):
    Ai1, Bi1 = Maclaurian(x, cutoff=20, max_terms=50)
    Ai2, Bi2 = AsimptotskaVrsta(x)
    Ai_maclaurian[i] = Ai1    
    Bi_maclaurian[i] = Bi1
    Ai_asymptotic[i] = Ai2        
    Bi_asymptotic[i] = Bi2
    


# Error calculation ____________________________________________________________
abs_err_ai_mac = np.abs(Ai_maclaurian - scipy_ai_values)
rel_err_ai_mac = np.abs(Ai_maclaurian - scipy_ai_values) / scipy_ai_values
abs_err_ai_asy = np.abs(Ai_asymptotic - scipy_ai_values)
rel_err_ai_asy = np.abs(Ai_asymptotic - scipy_ai_values) / scipy_ai_values

abs_err_bi_mac = np.abs(Ai_maclaurian - scipy_bi_values)
rel_err_bi_mac = np.abs(Ai_maclaurian - scipy_bi_values) / scipy_bi_values
abs_err_bi_asy = np.abs(Bi_asymptotic - scipy_bi_values)
rel_err_bi_asy = np.abs(Bi_asymptotic - scipy_bi_values) / scipy_bi_values

# Plotting of the functions ____________________________________________________

plt.figure(figsize=(12, 8))


plt.subplot(2, 2, 1)
plt.plot(x_values, scipy_ai_values, 'k--', label='Scipy Ai(x)', linewidth=2)
plt.plot(x_values, Ai_maclaurian, label='Maclaurian Ai(x)', linewidth=2)
plt.plot(x_values, Ai_asymptotic, label='Asymptotic Ai(x)', linewidth=2)

plt.xlabel('x')
plt.ylabel('Values')
plt.ylim((-0.6, 0.6))
plt.legend()
plt.title('Maclaurian Ai(x)')

plt.subplot(2, 2, 2)
plt.plot(x_values, rel_err_ai_mac, label='Rel. Err. Maclaurian', linewidth=2)
plt.plot(x_values, rel_err_ai_asy, label='Rel. Err. Asymptotic', linewidth=2)
plt.xlabel('x')
plt.ylabel('Relative Error')
plt.yscale('log')
plt.legend()
plt.title('Relative Error in Ai(x)')


plt.subplot(2, 2, 3)
plt.plot(x_values, scipy_bi_values, 'k--', label='Scipy Bi(x)', linewidth=2)
plt.plot(x_values, Bi_maclaurian, label='Maclaurian Bi(x)', linewidth=2)
plt.plot(x_values, Bi_asymptotic, label='Asymptotic Bi(x)', linewidth=2)

plt.xlabel('x')
plt.ylabel('Values')
plt.ylim((-0.6, 0.6))
plt.legend()
plt.title('Maclaurian Bi(x)')

plt.subplot(2, 2, 4)
plt.plot(x_values, rel_err_bi_mac, label='Rel. Err. Maclaurian', linewidth=2)
plt.plot(x_values, rel_err_bi_asy, label='Rel. Err. Asymptotic', linewidth=2)
plt.xlabel('x')
plt.ylabel('Relative Error')
plt.yscale('log')
plt.legend()
plt.title('Relative Error in Bi(x)')


plt.tight_layout()
plt.show()
