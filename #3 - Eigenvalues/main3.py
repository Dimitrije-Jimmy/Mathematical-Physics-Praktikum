import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import matplotlib.animation as animation
import sys
import os

from numpy import linalg as LA

def qr_algorithm_numpy(A, iterations=50):
    for i in range(iterations):
        # QR decomposition
        Q, R = LA.qr(A)
        A = np.dot(R, Q)

    eigenvalues = np.diag(A)
    return Q, A

def qr_householder(M):
    A = np.copy(M)
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)

    eigenvalues = np.diag(A)
    return Q, A, eigenvalues


def householder(M):
    A = np.copy(M)
    print(A.shape)
    m, n = A.shape
    H = np.eye(m)

    for k in range(n-2):
        x = A[k+1:, k] # Subvector of A[k+1:, k]
        
        v = np.zeros_like(x)
        v[0] = np.sign(x[0]) * LA.norm(x) # Householder vector

        v /= LA.norm(v) # Normalize v

        # Apply householder transformation to A
        A[k+1:, k:] -= 2*np.outer(v, np.dot(v, A[k+1:, k:]))
        A[:, k+1:] -= 2*np.outer(np.dot(A[:, k+1:], v), v)

        H[k+1:, k+1:] -= 2*np.outer(v, np.dot(v, H[k+1:, k+1:]))

        return H

def make_householder(a):
    #rescaling to v and sign for numerical stability
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.outer(v,v)
    return H


# Create a random 50x50 matrix
matrix_size = 50
matrix_size = 11
A = np.random.rand(matrix_size, matrix_size)

# Transform A into Hessenberg form
H = householder(A)
Q, A, eigenvalues = qr_householder(A)

print("Hessenberg matrix:")
print(H)
print("QR:")
print(Q)
print(A)
print(eigenvalues)


# Create a list to store snapshots of the matrix at each iteration
matrix_snapshots = []
def qr_householder2(M):
    A = np.copy(M)
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        matrix_snapshots.append(A)

        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)

    matrix_snapshots.append(A)   
    eigenvalues = np.diag(A)
    return Q, A, eigenvalues

A = np.random.rand(matrix_size, matrix_size)*10
Q, A, eigenvalues = qr_householder2(A)
print(len(matrix_snapshots))

# Perform Householder transformation and QR Algorithm iteratively
#for iteration in range(num_iterations):

    # Apply Householder transformation to A
    # Perform QR decomposition
    # Update A
    # Append a copy of A to matrix_snapshots
"""   
# Create a function to update the animation
def update(frame):
    print(frame)
    plt.clf()
    plt.subplot(121)
    plt.imshow(matrix_snapshots[frame], cmap='viridis')
    plt.title(f'Iteration {frame}')
    
    plt.subplot(122)
    cax = plt.matshow(matrix_snapshots[frame], cmap='viridis')
    plt.colorbar(cax, ax=plt.gca(), aspect=10)
"""

def init():
    im.set_data(matrix_snapshots[0])
    return im,

# Function to update the plot at each frame
def update(frame):
    im.set_data(matrix_snapshots[frame])
    ax1.set_title(f'Iteration {frame}')
    return im

# Create a figure with two subplots (matrix and colorbar)
fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

# Plot the initial matrix
im = ax1.imshow(matrix_snapshots[0], cmap='viridis')
ax1.set_title('Iteration 0')

# Create a colorbar in the first subplot
cbar = fig.colorbar(im, ax=ax1)

# Hide the colorbar in the first subplot
#cbar.remove()

# Set a custom frame interval (e.g., 10 milliseconds for a faster animation)
frame_interval = 10

# Create the animation
#fig, ax = plt.subplots()
#ani = animation.FuncAnimation(fig, update, frames=len(matrix_snapshots), repeat=False, interval=frame_interval)
ani = animation.FuncAnimation(fig, update, frames=len(matrix_snapshots), init_func=init, repeat=False, interval=frame_interval)

# Show the animation
plt.show()