import numpy as np
import matplotlib.pyplot as plt
import scipy.special

import matplotlib.animation as animation
import matplotlib.colors as color

import os

from numpy import linalg as LA

# Householder QR tridiagonal ___________________________________________________
def qr_householder(M):
    A = np.copy(M)
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
    #for i in range(n):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)

    eigenvalues = np.diag(A)

    # Sort eigenvalues and corresponding eigenvectors in descending order of absolute values
    sort_order = np.argsort(np.abs(eigenvalues))[::-1]  # Reverse the order
    eigenvalues = eigenvalues[sort_order]
    Q = Q[:, sort_order]

    return Q, A

def make_householder(a):
    #rescaling to v and sign for numerical stability
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.outer(v,v)
    return H


def zero_filter2(matrix, epsilon):
    A = np.copy(matrix)
    A[A < epsilon] = 0.0
    return A

def data_sort(diag, Q):
    diag_elements = np.diag(diag)
    vectors = np.copy(Q)
    output = []
    for coord, val in np.ndenumerate(diag_elements):
        output.append([val, vectors[coord[0]]])
    output.sort()

    return np.array(output)
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

def basis(q, n):
    return (2**n * np.math.factorial(n) * np.sqrt(np.pi))**(-1/2) * np.exp(-q**2/2) * scipy.special.eval_hermite(n, q)

def anharmonic(lamb, n, func):
    return lho(n) + lamb*func(n)


def plot_poly(x, vect):
    c = [vect[i]*basis(x, i) for i in range(len(vect))]
    return np.sum(c)


# Eigenvalue function __________________________________________________________
matrix_snapshots = []
def diagonalize(qr_function, matrix, tol=10**-15, maxiter=100000):
    tridiag = matrix
    matrix_snapshots.append(tridiag)
    
    s = np.eye(tridiag.shape[0])  # To store eigenvectors


    for i in range(maxiter):
        print(f"Iter: {i}")
        if i == maxiter - 1:
            #raise Warning("Maximum number of iterations exceeded")
            print("Maximum number of iterations exceeded")
        test =  np.abs(np.sum(np.abs(tridiag)) - np.sum(np.abs(np.diag(tridiag))))
        if test < 10**-10:
            print(i)
            break

        Q, R = qr_function(tridiag)
        tridiag = zero_filter2(np.matmul(Q.T, np.matmul(tridiag, Q)), tol)
        
        s = np.matmul(s, Q)

        matrix_snapshots.append(tridiag)

    return tridiag, s


lam = 1
data = anharmonic(lam, 100, q_matrix_single)
# data = anharmonic(lam, 10, q_matrix_double)
# data = anharmonic(lam, 10, q_matrix_quad)

diag, Q = diagonalize(qr_householder, data, tol=10**-5, maxiter=1000)
print(len(matrix_snapshots))


"""
# Plot matrix heat map animation
ani = ArtistAnimation(fig, img, interval=90, repeat=False, blit=True)
plt.title("Matrix diagonalization")
plt.axis("off")
plt.colorbar()
# ani.save("1.mp4", "ffmpeg", fps=20)
plt.show()
"""
# Create a colorbar in the first subplot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)  # Change the number of subplots to 1x3
plt.imshow(data, cmap='viridis')
plt.title('Začetno stanje')

plt.subplot(1, 2, 2)  # Use the middle subplot for the second image
plt.imshow(matrix_snapshots[-1], cmap='viridis')
plt.title('Končno stanje')


# Adjust the location and size of the colorbar
cbar = plt.colorbar(orientation='vertical', cax=plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7]))

#plt.tight_layout()
plt.show()


def init():
    im.set_data(matrix_snapshots[0])
    return im,

# Function to update the plot at each frame
def update(frame):
    print(frame)
    im.set_data(matrix_snapshots[frame])
    ax1.set_title(f'Iteration {frame+1}, for $\lambda = {lam}$')
    return im

# Create a figure with two subplots (matrix and colorbar)
fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

# Plot the initial matrix
im = ax1.imshow(matrix_snapshots[0], cmap='viridis')#, norm=color.LogNorm())#, norm=color.CenteredNorm(vcenter=0))
ax1.set_title('Iteration 0')

"""
ValueError: 'r_viridis' is not a valid value for cmap; supported values are 
'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 
'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2', 'PiYG_r', 'PuBu', 
'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 
'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 
'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 
'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 
'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 
'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 
'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 
'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 
'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 
'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 
'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gnuplot', 
'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 
'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 
'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 
'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 
'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 
'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 
'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 
'viridis', 'viridis_r', 'winter', 'winter_r'
"""
# importing datetime module for now()
import datetime
 
# using now() to get current time
current_time = str(datetime.datetime.now())
print(current_time)

current_time_fixed = current_time.split('.')[0]
current_time_fixed = current_time_fixed.replace('-', '_').replace(':', '.')




directory = os.path.dirname(__file__)+'\\Animations\\'
file_path = directory + f'{current_time_fixed}.mp4'
print(file_path)


# Create a colorbar in the first subplot
cbar = fig.colorbar(im, ax=ax1)

# Hide the colorbar in the first subplot
#cbar.remove()

# Set a custom frame interval (e.g., 10 milliseconds for a faster animation)
frame_interval = 20

# Create the animation
#fig, ax = plt.subplots()
#ani = animation.FuncAnimation(fig, update, frames=len(matrix_snapshots), repeat=False, interval=frame_interval)
ani = animation.FuncAnimation(fig, update, frames=len(matrix_snapshots), init_func=init, repeat=False, interval=frame_interval)
ani.save(file_path, writer='ffmpeg')#, "ffmpeg", fps=20)

# Show the animation
plt.show()
