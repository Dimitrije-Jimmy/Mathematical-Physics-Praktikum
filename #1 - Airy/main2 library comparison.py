import numpy as np
import matplotlib.pyplot as plt

# Primerjava implementacije Airy funkcij iz scipy in mpmath librarijev__________

from scipy.special import airy as scipy_airy
from mpmath import airyai as mpmath_airyai
from mpmath import airybi as mpmath_airybi


# Define the range of x values
x_values = np.linspace(-15, 10, 400)


# Initialize arrays to store the results
scipy_ai_values, scipy_bi_values = np.zeros_like(x_values), np.zeros_like(x_values)
mpmath_ai_values, mpmath_bi_values = np.zeros_like(x_values), np.zeros_like(x_values)

# Calculate Airy functions using scipy.special.airy and mpmath.airy
for i, x in enumerate(x_values):
    scipy_ai, scipy_aip, scipy_bi, scipy_bip = scipy_airy(x)
    mpmath_ai, mpmath_bi = mpmath_airyai(x), mpmath_airybi(x)
    scipy_ai_values[i], scipy_bi_values[i] = scipy_ai, scipy_bi
    mpmath_ai_values[i], mpmath_bi_values[i] = mpmath_ai, mpmath_bi

# Calculate absolute and relative errors
absolute_errors_ai = np.abs(scipy_ai_values - mpmath_ai_values)
relative_errors_ai = absolute_errors_ai / mpmath_ai_values

absolute_errors_bi = np.abs(scipy_bi_values - mpmath_bi_values)
relative_errors_bi = absolute_errors_bi / mpmath_bi_values


# Plotting the results _________________________________________________________
plt.figure(figsize=(10, 5))


plt.subplot(1, 2, 1)


plt.plot(x_values, relative_errors_ai, 'm-', alpha=0.5)
plt.scatter(x_values, relative_errors_ai, s=5, label='Ai(x)')

#plt.grid()
plt.xlabel('x')
plt.ylabel(r'$\vert Ai_{Scipy} - Ai_{Mpmath} \vert \;/\; Ai_{Mpmath}$')
plt.ylim((-2e-13, 2e-13))
plt.legend(loc='upper right')
plt.title('Relativna napaka Ai(x)')


plt.subplot(1, 2, 2)

plt.plot(x_values, relative_errors_ai, 'g-', alpha=0.5)
plt.scatter(x_values, relative_errors_ai, s=5, label='Bi(x)')

#plt.grid()
plt.xlabel('x')
plt.ylabel(r'$\vert Bi_{Scipy} - Bi_{Mpmath} \vert \;/\; Bi_{Mpmath}$')
plt.ylim((-2e-13, 2e-13))
plt.legend(loc='upper right')
plt.title('Relativna napaka Bi(x)')

plt.tight_layout()
plt.show()
plt.clf()
plt.close()
