import numpy as np
import matplotlib.pyplot as plt

import os

directory = os.path.dirname(__file__)


# Seed value
# Apparently you may use different seed values at each stage
seed_value= 10001
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)


# HISTOGRAM PARETO
def l(func_int, inverse_int_func, random_num, a, *args):
    return inverse_int_func((1-random_num) * func_int(a, args), args)
def P(l, mu):
    mu = mu[0]
    return 1 / ((1-mu) * l ** (mu-1))
def inverseP(y, mu):
    mu = mu[0]
    return (1 / ((1-mu) * y))**(1/(mu-1))
mu = 2
bins = 70
x = np.linspace(1, 5)
x_p = np.random.random(10000)
y = np.array([l(P, inverseP, x_k, 1, mu+1) for x_k in x_p])
w_x = mu * x ** (-mu - 1)
y_pareto = np.random.pareto(mu, size=len(x_p))+1
y_pareto = y_pareto[y_pareto < 5]
y = y[y < 5]


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.hist(y_pareto, bins=bins, density=True, label="numpy.random.pareto")
plt.plot(x, w_x, 'r', label="fit")

#plt.xlim([.9, 5])
plt.xlabel("x")
plt.ylabel(r" ${\rm d}p/{\rm d}x $")
plt.legend()

plt.subplot(1, 2, 2)

plt.hist(y, bins=bins, density=True, label="inverse CDF")
plt.plot(x, w_x, 'r', label="fit")

#plt.xlim([.9, 5])
plt.xlabel("x")
plt.ylabel(r" ${\rm d}p/{\rm d}x $")
plt.legend()

plt.tight_layout()
plt.show()
plt.close()
plt.clf()