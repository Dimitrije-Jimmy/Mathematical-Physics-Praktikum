"""
Finite differences method for finite and infinite potential well
"""
import numpy as np
from scipy.linalg import solve_banded


def fd(u, v, w, t, a, b):
    """
        x'' = u(t) + v(t) x + w(t) x'
        x(t[0]) = a, x(t[n-1]) = b

    u,v,w - arrays containing u(t), v(t), and w(t) values
    t     - array of n time values to determine x at
    a     - solution value at the left boundary: a = x(t[0])
    b     - solution value at the right boundary: b = x(t[n-1])
    """
    n = len(t)
    h = t[1] - t[0]
    if type(u) == int or type(u) == float:
        u = np.array([float(u)] * n)
    if type(v) == int or type(v) == float:
        v = np.array([float(v)] * n)
    if type(w) == int or type(w) == float:
        w = np.array([float(w)] * n)

    # Generate tridiagonal system
    A = np.array(-(1 + w[1:n] * h/2))
    A[-1] = 0.0

    C = np.array(-(1 - w[0:n-1] * h/2))
    C[0] = 0.0

    D = np.array(2.0 + h**2 * v)
    D[0] = D[n-1] = 1.0

    B = np.array(-h**2 * u)
    B[0] = a
    B[n-1] = b

    # Solve system
    for i in range(1, n):
        xmult = A[i-1]/D[i-1]
        D[i] = D[i] - xmult * C[i-1]
        B[i] = B[i] - xmult * B[i-1]

    x = np.zeros(n)
    x[n-1] = B[n-1]/D[n-1]

    for i in range(n-2, -1, -1):
        x[i] = (B[i] - C[i] * x[i+1])/D[i]

    return x


def gen_A(n, V, E, step):
    """
    Generate system of equations
    INPUT:
    n = dimension
    V = array of potentials
    E = energy
    step = step size

    OUTPUT:
    A = system matrix
    """
    A = np.zeros((n+1, n+1))
    A[0, 0] = 1
    A[n, n] = -2 + (V[n] - E) * step**2  # Not 100% sure if this is correct
    for i in range(1, n):
        A[i, i-1] = 1
        A[i, i] = -2 + (V[i] - E) * step**2  # Not 100% sure if this is correct
        A[i, i+1] = 1

    return A


def gen_b(n, V, E, step):
    """
    Generate solution vector
    INPUT:
    n = dimension
    V = array of potentials
    E = energy
    step = step size

    OUTPUT:
    b = solution vector
    """
    b = np.zeros(n+1)
    # Set boundary values (can be nonzero too)
    b[0] = 0
    b[-1] = 0
    for i in range(1, n-1):
        b[i] = (V[i] - E)*step**2

    return b


def gen_banded(A, b):
    ud = np.insert(np.diag(A, 1), 0, 0)
    d = np.diag(A)
    ld = np.insert(np.diag(A, -1), len(b) - 1, 0)
    ab = np.matrix([ud, d, ld])

    return np.array(ab)


def sol_banded(dim, V, E, step):
    A = gen_A(dim, V, E, step)
    b = gen_b(dim, V, E, step)
    ab = gen_banded(A, b)
    psi = solve_banded((1, 1), ab, b)

    return psi


# Poskus bolj splosno napisati
def gen_r(alpha, beta, P, R, h, n):
    """
    Generate vector R. For function given as:
    y''(x) = - P(x)y' - Q(x)y(x) - R(x)

    INPUTS:
    alpha: left boundary value
    beta: right boundary value
    h: step size
    n: dimension

    OUTPUT:
    r: vector R
    """
    r = np.zeros(n + 1)
    r[0] = h**2*R[0] - (1 - h/2 * P[0]) * alpha
    r[-1] = h**2*R[-1] - (1 + h/2 * P[-1]) * beta  # Tu je en plus sumljiv?
    for i in range(1, n-2):
        r[i] = h**2 * R[i]

    return r


def gen_z(P, Q, h, n):
    Z = np.zeros((n + 1, n + 1))
    Z[0, 0] = -(2 - h**2 * Q[0])
    Z[n, n] = -(2 - h**2 * Q[n-2])  # Oz. Q[-1]?
    Z[0, 1] = 1  # Debug snippet
    Z[n, -2] = 1
    for i in range(1, n):
        Z[i, i - 1] = 1 + h/2 * P[i - 1]  # Lower diagonal
        Z[i, i] = - (2 - h**2 * Q[i])     # Diagonal
        Z[i, i + 1] = 1 - h/2 * P[i + 1]  # Upper diagonal

    return Z


# ======================
def tridiag(a, b, c, n, k1=-1, k2=0, k3=1):
    if type(a) == int or type(a) == float:
        a = np.array([float(a)] * (n-1))
    if type(b) == int or type(b) == float:
        b = np.array([float(b)] * n)
    if type(c) == int or type(c) == float:
        c = np.array([float(c)] * (n-1))
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)


def matrika_neskoncen(n):
    A = np.zeros((n, n))
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:, 1:], 1)
    np.fill_diagonal(A, -2)
    A[0, 1] = 1
    A[-1, -2] = 1

    return A


def normalize(wavefunction):
    """Some sort of really rough normalization"""
    return wavefunction/np.max(wavefunction)


def fd_ipw(n, range, width):
    x = np.linspace(-range, range, n)
    h = x[1] - x[0]
    V = np.zeros(n)
    #diag = 1/h**2 * tridiag(1, -2, 1, n)
    diag = 1/h**2 * matrika_neskoncen(n)
    H = -1/2 * diag + np.diag(V)
    eigE, eigPsi = np.linalg.eigh(H)

    #return x, eigE, normalize(eigPsi)
    return x, eigE, eigPsi#/np.max(eigPsi)


def analytic(x, k):
    if k % 2 == 0:
        return np.sin(k*0.5*np.pi*x)
    else:
        return np.cos(k*0.5*np.pi*x)


def fd_fpw(n, range, depth):
    x = np.linspace(-range, range, n)
    h = x[1] - x[0]
    dim = len(x)
    pos = int(dim // 2.2)
    width = int(2 * (dim / 2 - pos))
    V_fpw = np.zeros(dim)
    V_fpw[:pos] = depth
    V_fpw[(pos + width):] = depth
    diag = 1/h**2 * tridiag(1, -2, 1, n)
    H = -1/2 * diag + np.diag(V_fpw)
    eigE, eigPsi = np.linalg.eigh(H)

    return x, eigE, normalize(eigPsi), V_fpw


def fd_fpw_32(n, range, depth):
    x = np.linspace(-range, range, n, dtype="float32")
    h = x[1] - x[0]
    pos = int(dim // 2.2)
    V_fpw = np.zeros(dim, dtype="float32")
    V_fpw[:pos] = depth
    V_fpw[(pos + width):] = depth
    diag = 1/h**2 * tridiag(1, -2, 1, n)
    H = -1/2 * diag + np.diag(V_fpw)
    H = np.float32(H)
    eigE, eigPsi = np.linalg.eigh(H)

    return x, eigE, normalize(eigPsi), V_fpw


def get_err(state, analytical):
    one = np.abs(state - analytical)
    two = np.abs(-state - analytical)
    if np.max(one) > np.max(two):
        return two
    else:
        return one


def fd_lho(n, range, depth):
    x = np.linspace(-range, range, n)
    h = x[1] - x[0]
    dim = len(x)
    pos = int(dim // 2.2)
    V_lho = x**2
    diag = 1/h**2 * tridiag(1, -2, 1, n)
    H = -1/2 * diag + np.diag(V_lho)
    eigE, eigPsi = np.linalg.eigh(H)
    return x, eigE, normalize(eigPsi), V_lho