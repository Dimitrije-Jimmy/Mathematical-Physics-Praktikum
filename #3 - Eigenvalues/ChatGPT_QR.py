import numpy as np
import time
import matplotlib.pyplot as plt
import numpy as np

from numpy import linalg as LA

# Your list of lists
data = [[12, -3, -2], [11.5, -2.6, -2.3], [12.5, -3.1, -1.9]]

# Convert the list of lists to a NumPy array for efficient element-wise operations
data_array = np.array(data)

# Calculate the element-wise absolute differences with the last sublist
differences = np.abs(data_array - data_array[-1])

# Sum the absolute differences along the rows
sum_of_differences = np.sum(differences, axis=1)

print("Sum of absolute differences:", sum_of_differences)


def jacobi_eigenvalue(A, tol=1e-9, max_iter=1000):
    n = A.shape[0]
    Q = np.eye(n)  # Initialize the orthogonal transformation matrix
    eigenvalues = np.zeros(n)
    
    for _ in range(max_iter):
        # Find the indices (p, q) of the maximum off-diagonal element
        p, q = np.unravel_index(np.argmax(np.abs(A - np.diag(np.diag(A))), axis=None), A.shape)
        
        if abs(A[p, q]) < tol:
            break  # Convergence criteria met
        
        # Calculate the rotation angle theta
        if A[p, p] == A[q, q]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan(2 * A[p, q] / (A[p, p] - A[q, q]))
        
        # Construct the rotation matrix J
        J = np.eye(n)
        J[p, p] = J[q, q] = np.cos(theta)
        J[p, q] = -np.sin(theta)
        J[q, p] = np.sin(theta)
        
        # Update A and Q with the rotation
        A = np.dot(J.T, np.dot(A, J))
        Q = np.dot(Q, J)
    
    eigenvalues = np.diag(A)

    # Sort eigenvalues and corresponding eigenvectors in descending order of absolute values
    sort_order = np.argsort(np.abs(eigenvalues))[::-1]  # Reverse the order
    eigenvalues = eigenvalues[sort_order]
    Q = Q[:, sort_order]
    
    return eigenvalues, Q

# Example usage:
matrix = np.array([[4.0, 1.0, 1.0],
                  [1.0, 3.0, 1.0],
                  [1.0, 1.0, 2.0]])

matrix = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

matrix = np.random.rand(10, 10)

eigenvalues, eigenvectors = jacobi_eigenvalue(matrix)
print("Eigenvalues:", eigenvalues)
#print("Eigenvectors:")
#print(eigenvectors)

eigenvalues, eigenvectors = LA.eig(matrix)
print("Eigenvalues eig:", eigenvalues)
#print("Eigenvectors eig:")
#print(eigenvectors)

eigenvalues_eigh, eigenvectors = LA.eigh(matrix)
# Sort eigenvalues in descending order of absolute values
sort_order = np.argsort(np.abs(eigenvalues_eigh))[::-1]
eigenvalues_eigh_sorted = eigenvalues_eigh[sort_order]
print("Eigenvalues eigh:", eigenvalues_eigh_sorted)
#print("Eigenvectors eigh:")
#print(eigenvectors)

def qr_algorithm_numpy(A, iterations=50):
    for i in range(iterations):
        # QR decomposition
        Q, R = np.linalg.qr(A)
        A = np.dot(R, Q)

    eigenvalues = np.diag(A)
    return eigenvalues



def qr_decomposition(A):
    m, n = A.shape
    Q = np.eye(m)

    for k in range(n):
        # Compute the R matrix
        R = np.eye(m)
        R[k:, k:] = np.linalg.norm(A[k:, k])  # Use numpy.linalg.norm to calculate the norm

        # Update A and Q
        A = np.dot(R, A)
        Q = np.dot(Q, R.T)

    return Q, A

def qr_algorithm(A, iterations=50):
    for i in range(iterations):
        Q, R = qr_decomposition(A)
        A = np.dot(R, Q)

    eigenvalues = np.diag(A)
    return eigenvalues



A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)


eigenvalues1 = qr_algorithm_numpy(A)
eigenvalues2 = qr_algorithm(A)
print("Eigenvalues:", eigenvalues1)
print("Eigenvalues:", eigenvalues2)


def compare_runtimes(max_size, step, iterations=50):
    sizes = list(range(1, max_size + 1, step))
    np_linalg_times = []
    qr_algorithm_times = []

    for size in sizes:
        A = np.random.rand(size, size)

        # Measure time for np.linalg.qr
        np_linalg_start = time.time()
        np.linalg.qr(A)
        np_linalg_end = time.time()
        np_linalg_time = np_linalg_end - np_linalg_start
        np_linalg_times.append(np_linalg_time)

        # Measure time for the QR algorithm from scratch
        qr_algorithm_start = time.time()
        qr_algorithm(A, iterations=iterations)
        qr_algorithm_end = time.time()
        qr_algorithm_time = qr_algorithm_end - qr_algorithm_start
        qr_algorithm_times.append(qr_algorithm_time)

    # Plot the results
    plt.plot(sizes, np_linalg_times, label="np.linalg.qr")
    plt.plot(sizes, qr_algorithm_times, label="QR Algorithm (from scratch)")
    plt.xlabel("Matrix Size")
    plt.ylabel("Execution Time (s)")
    #plt.yscale('log')
    plt.legend()
    plt.title("Runtime Comparison")
    plt.show()

compare_runtimes(max_size=1000, step=10)

