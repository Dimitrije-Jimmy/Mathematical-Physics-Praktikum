import numpy as np

# Example matrix with some small values
matrix = np.array([[1.0, 0.0001, 2.0],
                  [0.0002, 3.0, 0.0003],
                  [0.0004, 4.0, 5.0]])


def zero_filter(matrix, epsilon):
    a = np.copy(matrix)
    for index_pairs, value in np.ndenumerate(a):
        if np.abs(value) < epsilon:
            a[index_pairs[0]][index_pairs[1]] = 0
    return a


epsilon = 0.001  # Define your epsilon threshold

a = zero_filter(matrix, epsilon)

print(a)

# Set values smaller than epsilon to zero
matrix[matrix < epsilon] = 0

print(matrix)
