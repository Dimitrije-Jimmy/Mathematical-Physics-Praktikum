import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm
import numpy.linalg as LA

from scipy.linalg import solve_banded


# Functions ____________________________________________________________________

# Kubični B-spline
def Bspline2(xsez, x, k):    
    dx = xsez[1] - xsez[0]

    if x <= xsez[k-2]:
        return 0
    elif xsez[k-2] <= x <= xsez[k-1]:
        return ((x - xsez[k-2])/dx)**3
    elif xsez[k-1] <= x <= xsez[k]:
        return ((x - xsez[k-2])/dx)**3 - 4*((x - xsez[k-1])/dx)**3
    elif xsez[k] <= x <= xsez[k+1]:
        return ((xsez[k-2] - x)/dx)**3 - 4*((xsez[k-1] - x)/dx)**3
    elif xsez[k+1] <= x <= xsez[k+2]:
        return ((xsez[k-2] - x)/dx)**3
    elif xsez[k+2] <= x:
        return 0
    else:
        print("why the fuck not work ", k)
            

def Bspline3(xsez, x, k):    
    dx = xsez[1] - xsez[0]

    if x[k] <= xsez[k-2]:
        return 0
    elif xsez[k-2] <= x[k] <= xsez[k-1]:
        return ((x[k] - xsez[k-2])/dx)**3
    elif xsez[k-1] <= x[k] <= xsez[k]:
        return ((x[k] - xsez[k-2])/dx)**3 - 4*((x[k] - xsez[k-1])/dx)**3
    elif xsez[k] <= x[k] <= xsez[k+1]:
        return ((xsez[k-2] - x[k])/dx)**3 - 4*((xsez[k-1] - x[k])/dx)**3
    elif xsez[k+1] <= x[k] <= xsez[k+2]:
        return ((xsez[k-2] - x[k])/dx)**3
    elif xsez[k+2] <= x[k]:
        return 0
    else:
        print("why the fuck not work ", k)


def Bspline(xsez, x):
    B = np.zeros((len(x), len(xsez)))

    dx = xsez[1] - xsez[0]

    for k in range(-1, len(xsez) - 2):
        mask_1 = (x <= xsez[k-2])
        mask_2 = (xsez[k-2] <= x) & (x <= xsez[k-1])
        mask_3 = (xsez[k-1] <= x) & (x <= xsez[k])
        mask_4 = (xsez[k] <= x) & (x <= xsez[k+1])
        mask_5 = (xsez[k+1] <= x) & (x <= xsez[k+2])
        mask_6 = (xsez[k+2] <= x)

        B[mask_1, k] = 0
        B[mask_2, k] = ((x[mask_2] - xsez[k-2])/dx)**3
        B[mask_3, k] = ((x[mask_3] - xsez[k-2])/dx)**3 - 4*((x[mask_3] - xsez[k-1])/dx)**3
        B[mask_4, k] = ((xsez[k-2] - x[mask_4])/dx)**3 - 4*((xsez[k-1] - x[mask_4])/dx)**3
        B[mask_5, k] = ((xsez[k-2] - x[mask_5])/dx)**3
        B[mask_6, k] = 0

    return B

def Bspline5(xsez, x, k):
    B = np.zeros(len(xsez))

    dx = xsez[1] - xsez[0]

    #for k in range(-1, len(xsez) - 2):
    for i in range(len(xsez)):
        if x[i] <= xsez[k-2]:
            B[i] = 0
        elif xsez[k-2] <= x[i] <= xsez[k-1]:
            B[i] = ((x[i] - xsez[k-2])/dx)**3
        elif xsez[k-1] <= x[i] <= xsez[k]:
            B[i] = ((x[i] - xsez[k-2])/dx)**3 - 4*((x[i] - xsez[k-1])/dx)**3
        elif xsez[k] <= x[i] <= xsez[k+1]:
            B[i] = ((xsez[k-2] - x[i])/dx)**3 - 4*((xsez[k-1] - x[i])/dx)**3
        elif xsez[k+1] <= x[i] <= xsez[k+2]:
            B[i] = ((xsez[k-2] - x[i])/dx)**3
        elif xsez[k+2] <= x[i]:
            B[i] = 0
        else:
            print("why the fuck not work ", k)

    return B


def matrikaA(n):
    A = np.zeros((n, n))
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:, 1:], 1)
    np.fill_diagonal(A, 4)
    A[0, 1] = 1
    A[-1, -2] = 1

    return A

def matrikaB(n):
    global D, a, b, dx
    dx = int(b-a)/n
    
    B = np.zeros((n, n))
    np.fill_diagonal(B[1:], 1)
    np.fill_diagonal(B[:, 1:], 1)
    np.fill_diagonal(B, -2)
    B[0, 1] = 1
    B[-1, -2] = 1

    B = 6*D*B/dx**2
    return B


def Ckoeficienti(n, f):
    global dt
    
    A = matrikaA(n)
    B = matrikaB(n)

    LHS = (A - 0.5*dt*B)
    RHS = (A + 0.5*dt*B)
    
    c = np.zeros((n, n))
    c[0] = np.matmul( LA.inv(A), f )
    
    for i in range(1, len(t)):
        part = np.matmul( LA.inv(LHS), RHS )
        c[i] = np.matmul( part, c[i-1] )
        
    return c
    

# Parameters ___________________________________________________________________
a, b, n, m = 0., 1., 100, 2
L, D = 1.0, 0.05
sigma = L/4.
N, M = n, m

tk = 20
tn = 100
tn = n

x = np.linspace(0, L, n)    # Define x domain
u0 = np.exp(-((x - L/2.)**2.)/(sigma**2.))

x2 = np.linspace(0, 2*L*(1. - (1./(2.*N))), N)
#u0 = np.exp(-((x2 - L/2.)**2.)/(sigma**2.)) - np.exp(-((x2 - 3.*L/2.)**2.)/(sigma**2.))  # Dirichletov robni
#u0 = np.sin(np.pi*x)
#u0hat = np.fft.fft(u0)
#x = x2

t = np.linspace(0, tk, tn)
dt = t[1] - t[0]

#n = 4
A = matrikaA(n)
B = matrikaB(n)

LHS = (A - 0.5*dt*B)
RHS = (A + 0.5*dt*B)
#print(LHS)
#print(RHS)

f = u0

c = Ckoeficienti(n, f)
B = Bspline(x, x)

#print(c)
#print(B)
print(c.shape)
print(B.shape)

Txt = np.zeros((len(t), len(x)))
#u = np.zeros((tn, n+2))
"""
for i, tval in enumerate(t):
    print(i)
    #for k in range(-1, n+1):
    c = Ckoeficienti(n, f)
    B = Bspline(x, x)
    v = sum( np.dot(c, B) )
    
    Txt[i] = v

"""
"""
for i, tval in enumerate(t):
    print(i)
    c = Ckoeficienti(n, f)
    for j, xval in enumerate(x):
        v = 0. 
        
        #for k in range(-1, n+1):
        for k in range(len(c)):
            B = Bspline2(x, xval, k)
            v += c[i, k] * B
        
        #print(c.shape)
        Txt[i, j] = v
"""
"""
for i, tval in enumerate(t):
    print(i)
    #for k in range(-1, n+1):
    c = Ckoeficienti(n, f)
    B = Bspline(x, x)
    #v = sum( np.matmul(c, B) )
    v = 0.
    for j in range(len(c)):
        v += np.sum( np.dot(c[j], B[j]) )
    
    Txt[i] = v
"""

   
"""
def solve_tridiagonal(A, B, dt, c_n):
    #n = len(c_n)
    #ab = np.vstack([B[-1, :], A, B[1, :]])
    
    AB = [B[-1, :], A.diagonal(), B[1, :]]
    # Create the right-hand side of the system
    rhs = np.dot(A + dt/2 * B, c_n)
    
    # Solve the tridiagonal system using solve_banded
    u[1:-1,j+1] = solve_banded((1,1), AB, rhs)
    #u_n1 = solve_banded((1, 1), AB, rhs)
    
    #return u_n1

# k <-- dt
# K <-- D
# h <-- dx
# h2 <-- dx^2

for j in range(tn):
    #AB = np.array([B[-1, :], A.diagonal(), B[1, :]])
    lower_diag = np.hstack([0, B[1:, 0]])
    upper_diag = np.hstack([B[:-1, -1], 0])
    AB = np.array([lower_diag, A.diagonal(), upper_diag])
    AB = np.vstack([B[-1, :], A, B[1, :]])
    
    rhs = np.zeros( n - 1, float )
    rhs = dt*D*(u[0:-2,j] + u[2:,j]) - (2*dt*D/dx**2) * u[1:-1,j]
    rhs[0]  = rhs[0]  + dt*D*u[0,j+1]
    rhs[-1] = rhs[-1] + dt*D*u[n-1,j+1]

    print(AB.shape, rhs.shape)

    uu = solve_banded((1,1), AB, rhs)
    u[1:-1,j+1] = uu
    #u[1:-1,j+1] = tridiagonal.solve( A, D, C, B )
"""


Txt = np.zeros((len(t), len(x)))

C = Ckoeficienti(n, f)
Bk = Bspline(x, x)
for i in range(len(t)):
    #print(len(c[i]))
    for k in range(-1, N-2):
        Txt[i] += c[i, k]*Bspline5(x, x, k)
    #print(Bk[i])
    #print(C[i])
    #print(len(np.dot(c[i], Bk)) )
    #Txt[i] += sum(np.dot(c[i], Bk[i]))

# Plotting _____________________________________________________________________

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#plt.set_cmap('jet_r')

X, T = np.meshgrid(x, t)
surf = ax.plot_surface(X, T, Txt, cmap='jet')    # Dirichletov

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$')
ax.set_zlabel(r'T($x, t$)')
ax.set_title(r'T($x, t$)')

# Add colorbar
fig.colorbar(surf, ax=ax, label='Temperature')

plt.show()