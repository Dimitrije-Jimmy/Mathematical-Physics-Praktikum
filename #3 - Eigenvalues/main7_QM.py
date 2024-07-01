import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import os

import matplotlib.animation as animation
import matplotlib.colors as color

from numpy import linalg as LA

#plt.ion()

# Quantum Mechanics ____________________________________________________________
def lho(n):
    return np.diag([i + 1/2 for i in range(0, n)])

def delta(i, j):
    return int(i == j)

def q_matrix_single(n):
    matrix = np.zeros((n, n))
    for ind, val in np.ndenumerate(matrix):
        i, j = ind
        matrix[i][j] = 0.5 * np.sqrt(i + j + 1) * delta(np.abs(i - j), 1)

    return np.matmul(matrix, np.matmul(matrix, np.matmul(matrix, matrix)))

def q_matrix_double(n):
    matrix = np.zeros((n, n))
    for ind, val in np.ndenumerate(matrix):
        i, j = ind
        a = np.sqrt(j * (j - 1)) * delta(i, j - 2)
        b = (2 * j + 1) * delta(i, j)
        c = np.sqrt((j + 1) * (j + 2)) * delta(i, j + 2)
        matrix[i][j] = 0.5 * (a + b + c)

    return np.matmul(matrix, matrix)

def q_matrix_quad(n):
    matrix = np.zeros((n, n))
    for ind, val in np.ndenumerate(matrix):
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
    return lho(n) + lamb*func(n)


def plot_poly(x, vect):
    c = [vect[i]*basis(x, i) for i in range(len(vect))]
    return np.sum(c)


# Eigenvalue function __________________________________________________________
def qr_numpy(A, iterations=50):
    for i in range(iterations):
        # QR decomposition
        Q, R = np.linalg.qr(A)
        A = np.dot(R, Q)

    eigenvalues = np.diag(A)
    return eigenvalues, Q

# Plotting _____________________________________________________________________
def plot_poly(x, vect):
    c = [vect[i]*basis(x, i) for i in range(len(vect))]
    return np.sum(c)


def arrayize(array, function, *args):
    return np.array([function(x, *args) for x in array])


plt.figure(figsize=(8, 5))

cmap = plt.get_cmap('viridis')
barve = [cmap(value) for value in np.linspace(0, 1, 10)]


# Plot osnovnih vezanih stanj
#plt.subplot(1, 2, 1)
plt.title("Prvih 10 vezanih stanj za nemoten harmonski oscilator")
x = np.linspace(-5, 5, 200)
plt.plot(x, x**2, label="V(x)", color='black')#)
plt.ylim(-1, 20)
for i in range(10):
    plt.plot(x, basis(x, i) + 2*i, color=barve[i])
plt.xlabel(r"$x$")
plt.ylabel(r"$|n\rangle$")
plt.legend()

plt.show()
plt.clf()
plt.close()




lam = 1
data = anharmonic(lam, 10, q_matrix_single)
# data = anharmonic(lam, 10, q_matrix_double)
# data = anharmonic(lam, 10, q_matrix_quad)

#diag, Q = qr_numpy(data)
diag, Q = LA.eigh(data)
print(diag)

# Plot lastnih stanj za motnjo lambda
plt.figure(figsize=(15, 8))

#lam = 1
#plt.subplot(1, 2, 2)
plt.title(r"Prvih 10 lastnih stanj za anharmonski oscilator z $\lambda = {}$".format(lam))
x = np.linspace(-5, 5, 200)
plt.plot(x, x**2 + lam*x**4, label="V(x)", color='black')#, color="#FFA0FD")
plt.ylim(-1, 20)

i = 0
#for vec in range(Q.shape[1]):
for vec in range(10):
    plt.plot(x, arrayize(x, plot_poly, Q[:, vec]) + 2*i, color=barve[i])#, color="#{}".format(color_vec[i]))
    i += 1

plt.xlabel(r"$x$")
plt.ylabel(r"$|n\rangle$")
plt.subplots_adjust(bottom=0.21)
plt.legend()
#plt.show()
plt.clf()
plt.close()


# Different q __________________________________________________________________
lam = 0.5
data1 = anharmonic(lam, 50, q_matrix_single)
data2 = anharmonic(lam, 50, q_matrix_double)
data3 = anharmonic(lam, 50, q_matrix_quad)
diag1, Q1 = LA.eigh(data1)
diag2, Q2 = LA.eigh(data2)
diag3, Q3 = LA.eigh(data3)
print(np.around(diag1[:10], 4))
print(np.around(diag2[:10], 4))
print(np.around(diag3[:10], 4))

# Vectors ______________________________________________________________
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.imshow(Q1)
plt.title(f"$[q]^4$, $\lambda = {lam}$")

plt.subplot(1, 3, 2)
plt.imshow(Q2)
plt.title(f"$[q^2]^2$, $\lambda = {lam}$")

plt.subplot(1, 3, 3)
plt.imshow(Q3)
plt.title(f"$[q^4]$, $\lambda = {lam}$")

plt.tight_layout()
#plt.show()
plt.clf()
plt.close()


# lastna stanja ________________________________________________________
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.title(f"$[q]^4$, $\lambda = {lam}$")
x = np.linspace(-5, 5, 200)
plt.plot(x, x**2 + lam*x**4, label="V(x)", color='black')#, color="#FFA0FD")
plt.ylim(-1, 20)

i = 0
#for vec in range(Q.shape[1]):
for vec in range(10):
    plt.plot(x, arrayize(x, plot_poly, Q1[:, vec]) + 2*i, color=barve[i])#, color="#{}".format(color_vec[i]))
    i += 1

plt.xlabel(r"$x$")
plt.ylabel(r"$|n\rangle$")
plt.subplots_adjust(bottom=0.21)
plt.legend()


plt.subplot(1, 3, 2)
plt.title(f"$[q^2]^2$, $\lambda = {lam}$")
x = np.linspace(-5, 5, 200)
plt.plot(x, x**2 + lam*x**4, label="V(x)", color='black')#, color="#FFA0FD")
plt.ylim(-1, 20)

i = 0
#for vec in range(Q.shape[1]):
for vec in range(10):
    plt.plot(x, arrayize(x, plot_poly, Q2[:, vec]) + 2*i, color=barve[i])#, color="#{}".format(color_vec[i]))
    i += 1

plt.xlabel(r"$x$")
plt.ylabel(r"$|n\rangle$")
plt.subplots_adjust(bottom=0.21)
plt.legend()

plt.subplot(1, 3, 3)
plt.title(f"$[q^4]$, $\lambda = {lam}$")
x = np.linspace(-5, 5, 200)
plt.plot(x, x**2 + lam*x**4, label="V(x)", color='black')#, color="#FFA0FD")
plt.ylim(-1, 20)

i = 0
#for vec in range(Q.shape[1]):
for vec in range(10):
    plt.plot(x, arrayize(x, plot_poly, Q2[:, vec]) + 2*i, color=barve[i])#, color="#{}".format(color_vec[i]))
    i += 1

plt.xlabel(r"$x$")
plt.ylabel(r"$|n\rangle$")
plt.subplots_adjust(bottom=0.21)
plt.legend()

plt.tight_layout()
plt.show()
plt.clf()
plt.close()


import sys
sys.exit()
# Test _________________________________________________________________________
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.title("Harmonski oscilator")
x = np.linspace(-5, 5, 200)
plt.plot(x, x**2, color='black')#)
plt.ylim(-1, 20)
for i in range(10):
    plt.plot(x, basis(x, i) + 2*i, label="V(x)", color=barve[i])
plt.xlabel(r"$x$")
plt.ylabel(r"$|n\rangle$")


plt.subplot(2, 2, 2)
plt.title(f"$[q]^4$, $\lambda = {lam}$")
x = np.linspace(-5, 5, 200)
plt.plot(x, x**2 + lam*x**4, label="V(x)", color='black')#, color="#FFA0FD")
plt.ylim(-1, 20)

i = 0
#for vec in range(Q.shape[1]):
for vec in range(10):
    plt.plot(x, arrayize(x, plot_poly, Q1[:, vec]) + 2*i, color=barve[i])#, color="#{}".format(color_vec[i]))
    i += 1

plt.xlabel(r"$x$")
plt.ylabel(r"$|n\rangle$")
plt.subplots_adjust(bottom=0.21)
plt.legend()


plt.subplot(2, 2, 3)
plt.title(f"$[q^2]^2$, $\lambda = {lam}$")
x = np.linspace(-5, 5, 200)
plt.plot(x, x**2 + lam*x**4, label="V(x)", color='black')#, color="#FFA0FD")
plt.ylim(-1, 20)

i = 0
#for vec in range(Q.shape[1]):
for vec in range(10):
    plt.plot(x, arrayize(x, plot_poly, Q2[:, vec]) + 2*i, color=barve[i])#, color="#{}".format(color_vec[i]))
    i += 1

plt.xlabel(r"$x$")
plt.ylabel(r"$|n\rangle$")
plt.subplots_adjust(bottom=0.21)
plt.legend()

plt.subplot(2, 2, 4)
plt.title(f"$[q^4]$, $\lambda = {lam}$")
x = np.linspace(-5, 5, 200)
plt.plot(x, x**2 + lam*x**4, label="V(x)", color='black')#, color="#FFA0FD")
plt.ylim(-1, 20)

i = 0
#for vec in range(Q.shape[1]):
for vec in range(10):
    plt.plot(x, arrayize(x, plot_poly, Q2[:, vec]) + 2*i, color=barve[i])#, color="#{}".format(color_vec[i]))
    i += 1

plt.xlabel(r"$x$")
plt.ylabel(r"$|n\rangle$")
plt.subplots_adjust(bottom=0.21)
plt.legend()

plt.tight_layout()
plt.show()
plt.clf()
plt.close()

# Animation ____________________________________________________________________
sys.exit()

#plt.imshow(data)
#plt.show()

lamb_sez = np.linspace(0, 1, 200)

matrix_snapshots = []
for i in range(len(lamb_sez)):
    data = anharmonic(lamb_sez[i], 30, q_matrix_single)

    diag, Q = LA.eigh(data)
    matrix_snapshots.append(Q)


def init():
    x = np.linspace(-5, 5, 200)
    plt.plot(x, x**2, color='black')#)
    plt.ylim(-1, 20)
    for i in range(10):
        plt.plot(x, basis(x, i) + 2*i, color=barve[i])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$|n\rangle$")


# Function to update the plot at each frame
def update(frame):
    global barve

    print(frame)
    #im.data(matrix_snapshots[frame])
    plt.clf()

    plt.title("Prvih 10 lastnih stanj za anharmonski oscilator z $\lambda = {}$".format(np.around(lamb_sez[frame], 3)))
    x = np.linspace(-5, 5, 200)
    plt.plot(x, x**2 + lamb_sez[frame]*x**4, color='black')#, color="#FFA0FD")
    plt.ylim(-1, 20)

    Q = matrix_snapshots[frame]
    i = 0
    #for vec in range(Q.shape[1]):
    for vec in range(10):
        plt.plot(x, arrayize(x, plot_poly, Q[:, vec]) + 2*i, color=barve[i])
        i += 1

    plt.xlabel(r"$x$")
    plt.ylabel(r"$|n\rangle$")
    plt.subplots_adjust(bottom=0.21)


# Create a figure with two subplots (matrix and colorbar)
fig = plt.figure(figsize=(8, 5))


# To not override my animations
import datetime
current_time = str(datetime.datetime.now())
print(current_time)
current_time_fixed = current_time.split('.')[0]
current_time_fixed = current_time_fixed.replace('-', '_').replace(':', '.')


directory = os.path.dirname(__file__)+'\\Animations\\'
file_path = directory + f'QM_{current_time_fixed}.mp4'
print(file_path)


# Set a custom frame interval (e.g., 10 milliseconds for a faster animation)
frame_interval = 30

# Create the animation
#fig, plt = plt.subplots()
#ani = animation.FuncAnimation(fig, update, frames=len(matrix_snapshots), repeat=False, interval=frame_interval)
ani = animation.FuncAnimation(fig, update, frames=len(matrix_snapshots), repeat=False, interval=frame_interval)
ani.save(file_path, writer='ffmpeg')#, "ffmpeg", fps=20)

# Show the animation
plt.show()
