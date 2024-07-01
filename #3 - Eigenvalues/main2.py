import numpy as np
import matplotlib.pyplot as plt
import sys

from numpy import linalg as LA

import scipy.special


def linearni_harmonski_oscilator(n):
    return np.diag([i + 1/2 for i in range(0, n)])

def delta(i, j):
    return int(i == j)

def q_single(n):
    matrix = np.zeros((n, n))
    for ind, _ in np.ndenumerate(matrix):
        print(ind)
        i, j = ind
        matrix[i][j] = 0.5 * np.sqrt(i + j + 1) * delta(np.abs(i - j), 1)

    return np.matmul(matrix, np.matmul(matrix, np.matmul(matrix, matrix)))


def q_double(n):
    matrix = np.zeros((n, n))
    for ind, _ in np.ndenumerate(matrix):
        i, j = ind
        a = np.sqrt(j * (j - 1)) * delta(i, j - 2)
        b = (2 * j + 1) * delta(i, j)
        c = np.sqrt((j + 1) * (j + 2)) * delta(i, j + 2)
        matrix[i][j] = 0.5 * (a + b + c)

    return np.matmul(matrix, matrix)


def q_quad(n):
    matrix = np.zeros((n, n))
    for ind, _ in np.ndenumerate(matrix):
        i, j = ind
        prefac = 1/(2**4) * np.sqrt((2**i * np.math.factorial(i))/(2**j * np.math.factorial(j)))
        a = delta(i, j + 4)
        b = 4*(2 * j + 3) * delta(i, j + 2)
        c = 12*(2*j**2 + 2*j + 1) * delta(i, j)
        d = 16*j*(2*j**2 - 3*j + 1) * delta(i, j - 2)
        e = 16*j*(j**3 - 6*j**2 + 11*j - 6) * delta(i, j - 4)
        matrix[i][j] = prefac * (a + b + c + d + e)

    return matrix

def basis(q, n):
    return (2**n * np.math.factorial(n) * np.sqrt(np.pi))**(-1/2) * np.exp(-q**2/2) * scipy.special.eval_hermite(n, q)


def anharmonic(lamb, n, func):
    return linearni_harmonski_oscilator(n) + lamb*func(n)


def plot_poly(x, vect):
    c = [vect[i]*basis(x, i) for i in range(len(vect))]
    return np.sum(c)