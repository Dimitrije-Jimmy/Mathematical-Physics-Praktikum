import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scs
from scipy.sparse.linalg import spsolve
from scipy.sparse import block_diag
from mpl_toolkits.mplot3d import Axes3D

from main1_functions import *

import os
dir_path = os.path.dirname(os.path.realpath(__file__))+'\\Images\\Results\\'
print(dir_path)

# Functions ____________________________________________________________________

def analytic(xi, t):
    # xi and t should be meshgrid
    return np.sin(np.pi*np.cos(xi + t))

def valovna(xi, k):
    return np.exp(1j*k*xi) / np.sqrt(2*np.pi)


def analiticni_koef(k, t):
    # scs.jv calculates for positive and negative k
    return np.sin(0.5*np.pi*k) * np.exp(1j*k*t) * scs.jv(k, np.pi)


#def diference()


# Parameters ___________________________________________________________________

N = 1000
M_max = 3
N_max = 3

k = np.linspace(-N/2, N/2, 2*N)

t = np.linspace(0, 1, N)
xi = np.linspace(0, 2*np.pi, N)
XI, T = np.meshgrid(xi, t) #za plt.countourf
#X = R * np.cos(PHI)
#Y = R * np.sin(PHI)

vf0 = analytic(xi, 0) # začetni pogoj
