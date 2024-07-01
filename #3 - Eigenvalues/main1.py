import numpy as np
import matplotlib.pyplot as plt
import sys

from numpy import linalg as LA

import scipy.special

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num GPUs Available: ", tf.config.list_physical_devices('XLA_GPU'))


from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())

# Your output is probably something like ['/device:CPU:0']
# It should be ['/device:CPU:0', '/device:GPU:0']


#A je mxn matrika
#Q je mxm ortogonalna
#R=rezultat, koncna A je 'zgornje' trikotna


def qr(M):
    A = np.copy(M)
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A

def make_householder(a):
    #rescaling to v and sign for numerical stability
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.outer(v,v)
    return H


def qr_givens(M):
    A = np.copy(M)
    m, n = A.shape
    Q = np.eye(m)
    for j in range(n - (m == n)):
        for i in range(j+1,m):
            r=np.hypot(A[j,j],A[i,j])
            c=A[j,j]/r
            s=A[i,j]/r
            givensRot = np.array([[c, s],[-s,  c]])
            A[[j,i],j:] = np.dot(givensRot, A[[j,i],j:])
            Q[[j,i],:] = np.dot(givensRot, Q[[j,i],:])
    return Q.T, A

def trid_householder(M):
    A = np.copy(M)
    m, n = A.shape
    if ( m != n):
        print("need quadratic symmetric matrix")
        sys.exit(1)
    Q = np.eye(m)
    for i in range(m - 2):
        H = np.eye(m)
        H[i+1:, i+1:] = make_householder(A[i+1:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
        A = np.dot(A,H)
    return Q, A



def linearni_harmonski_oscilator(n):
    return np.diag([i + 1/2 for i in range(0, n)])

def delta(i, j):
    return int(i == j)