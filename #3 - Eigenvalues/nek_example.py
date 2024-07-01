import matplotlib
import tensorflow as tf

# Let’s see how we can compute the eigen vectors and values from a matrix
e_matrix_A = tf.random.uniform([2, 2], minval=3, maxval=10, dtype=tf.float32, name="matrixA")
print("Matrix A: \n{}\n\n".format(e_matrix_A))

# Calculating the eigen values and vectors using tf.linalg.eigh, if you only want the values you can use eigvalsh
eigen_values_A, eigen_vectors_A = tf.linalg.eigh(e_matrix_A)
print("Eigen Vectors: \n{} \n\nEigen Values: \n{}\n".format(eigen_vectors_A, eigen_values_A))

# Now lets plot our Matrix with the Eigen vector and see how it looks
Av = tf.tensordot(e_matrix_A, eigen_vectors_A, axes=0)
vector_plot([tf.reshape(Av, [-1]), tf.reshape(eigen_vectors_A, [-1])], 10, 10)

"""
Matrix A:
[[5.450138 9.455662]
 [9.980919 9.223391]]

Eigen Vectors:
[[-0.76997876 -0.6380696 ]
 [ 0.6380696 -0.76997876]]

Eigen Values:
[-2.8208985 17.494429 ]
"""

# Lets us multiply our eigen vector by a random value s and plot the above graph again to see the rescaling
sv = tf.multiply(5, eigen_vectors_A)
vector_plot([tf.reshape(Av, [-1]), tf.reshape(sv, [-1])], 10, 10)


# Creating a matrix A to find it’s decomposition
eig_matrix_A = tf.constant([[5, 1], [3, 3]], dtype=tf.float32)
new_eigen_values_A, new_eigen_vectors_A = tf.linalg.eigh(eig_matrix_A)
print("Eigen Values of Matrix A: {} \n\nEigen Vector of Matrix A: \n{}\n".format(new_eigen_values_A, new_eigen_vectors_A))

# calculate the diag(lamda)
diag_lambda = tf.linalg.diag(new_eigen_values_A)
print("Diagonal of Lambda: \n{}\n".format(diag_lambda))

# Find the eigendecomposition of matrix A
decomp_A = tf.tensordot(tf.tensordot(eigen_vectors_A, diag_lambda, axes=1), tf.linalg.inv(new_eigen_vectors_A), axes=1)
print("The decomposition Matrix A: \n{}".format(decomp_A))

"""
Eigen Values of Matrix A: 
[0.8377223 7.1622777]

Eigen Vector of Matrix A:
[[-0.5847103 0.81124216]
 [ 0.81124216 0.5847103 ]]

Diagonal of Lambda:
[[0.8377223 0. ]
 [0. 7.1622777]]

The decomposition Matrix A:
[[-3.3302479 -3.195419 ]
 [-4.786382 -2.7909322]]
"""


# In section 2.6 we manually created a matrix to verify if it is symmetric, 
#  but what if we don’t know the exact values and want to create a random symmetric matrix
new_matrix_A = tf.Variable(tf.random.uniform([2,2], minval=1, maxval=10, dtype=tf.float32))

# to create an upper triangular matrix from a square one
X_upper = tf.linalg.band_part(new_matrix_A, 0, -1)
sym_matrix_A = tf.multiply(0.5, (X_upper + tf.transpose(X_upper)))
print("Symmetric Matrix A: \n{}\n".format(sym_matrix_A))

# create orthogonal matrix Q from eigen vectors of A
eigen_values_Q, eigen_vectors_Q = tf.linalg.eigh(sym_matrix_A)
print("Matrix Q: \n{}\n".format(eigen_vectors_Q))

# putting eigen values in a diagonal matrix
new_diag_lambda = tf.linalg.diag(eigen_values_Q)
print("Matrix Lambda: \n{}\n".format(new_diag_lambda))

sym_RHS = tf.tensordot(tf.tensordot(eigen_vectors_Q, new_diag_lambda, axes=1), tf.transpose(eigen_vectors_Q), axes=1)
predictor = tf.reduce_all(tf.equal(tf.round(sym_RHS), tf.round(sym_matrix_A)))
def true_print(): print("It WORKS. \nRHS: \n{} \n\nLHS: \n{}".format(sym_RHS, sym_matrix_A))
def false_print(): print("Condition FAILED. \nRHS: \n{} \n\nLHS: \n{}".format(sym_RHS, sym_matrix_A))

tf.cond(predictor, true_print, false_print)

"""
Symmetric Matrix A:
[[4.517448 3.3404353]
 [3.3404353 7.411926 ]]

Matrix Q:
[[-0.8359252 -0.5488433]
 [ 0.5488433 -0.8359252]]

Matrix Lambda:
[[2.3242188 0. ]
 [0. 9.605155 ]]

It WORKS.

RHS:
[[4.5174475 3.340435 ]
 [3.340435 7.4119253]]

LHS:
[[4.517448 3.3404353]
 [3.3404353 7.411926 ]]
"""