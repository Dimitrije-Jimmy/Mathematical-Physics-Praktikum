# Pioneer of Success - Python code Crank Nicolson Method for 1D Unsteady Heat Conduction

"""
# Crank Nicolson je un metuljček pattern

\partial T / \partial t = \alpha \partial^2 T / \partial x^2

0 < x < l;  t > 0

T(x, 0) = 0
T(0, t) = 100
T(l, t) = 0

T(x, t) = ?

n <-- \Delta t - time step
h <-- \Delta x - spac step

with Dirichlet boundary conditions T(x0, t) = 0, T(xL, t) = 0
and initial condition u(x, 0) = 4*x - 4*x**2


Forward Time:
dT/dt = (T(x, t+n) - T(x, t)) / n

Average Space:
d^2T/dx^2 = 0.5*( (T(x+h, t+n) - 2T(x, t+n) + T(x-h, t+n)) / x^2 +  (T(x+h, t) - 2T(x, t) + T(x-h, t)) / h^2  )

=> ( T(x, t+n) - T(x, t) ) / n = 0.5*(  )

Vectorized:
A T_{j+1} = B T_j + b_j + b_{j+1} 
T_{j+1} = A^-1 * (B T_j + b_j + b_{j+1}) 

Sam nardit rabiš tridiagonalno matriko
"""

# Crank Nicolson Method for Unstead Heat Equation
# Shared online by Pioneer of Success
# Email: pioneerofsuccess2020@gmail.com
# Disclaimer: The codes are developed taking help from various online free resources,
#  we do not claim copyright for this code

# Required Libraries
# numpy library for matrix and vectors
import numpy as np
# library for mathematical functions
import math

# Libraries required for plotting and manipulations
#%matplotlib inline
import matplotlib.pyplot as plt     # side-stepping mpl backend
import warnings
warnings.filterwarnings("ignore")

N = 10      # number of divisions in spatial direction
Nt = 100    # number of divisions in time direction
h = 1/N     # each division in space - dx
k = 1/Nt    # each division in time - dt
r = k/(h*h) # coefficient
time_steps = 15 # code will be solved for 15 time steps
time = np.arange(0, (time_steps+.5)*k, k)   # linear array creation for time
x = np.arange(0, 1.0001, h) # linear array for space

X, Y = np.meshgrid(x, time)
fig = plt.figure()

plt.plot(X, Y, 'ro');
plt.plot(x, 0*x, 'bo', label='Initial Condition');
plt.plot(np.ones(time_steps+1), time, 'go', label='Boundary Condition');
plt.plot(x, 0*x, 'bo');
plt.plot(0*time, time, 'go')

plt.xlim((-0.02, 1.02))
plt.xlabel('x')
plt.ylabel('time [ms]')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(r'Discrete Grid $\Omega_h,$ h= %s, k%s'%(h, k), fontsize=12, y=1.08)
plt.tight_layout()
plt.show();


# This part defines the initial and boundary conditions
w = np.zeros((N+1, time_steps+1))
b = np.zeros(N-1)

# Initial Condition
for i in range(1, N):
    w[i, 0] = 4*x[i] - 4*x[i]*x[i]
    # if x[i] > 0.5:
    #   w[i, 0] = 2*(1 - x[i])
    
# Boundary Condition
for k in range(0, time_steps):
    w[0, k] = 0
    w[N, k] = 0
    
fig = plt.figure(figsize=(8, 4))
plt.plot(x, w[:, 0], 'o:', label='Initial Condition')
plt.plot(x[[0, N]], w[[0, N], 0], 'go', label='Boundary Condition t[0] = 0')
#plt.plot(x[N], w[N, 0], 'go')

plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('x')
plt.ylabel('w')
plt.legend(loc='best')
plt.show()


# Defining matrixes
A = np.zeros((N-1, N-1))
B = np.zeros((N-1, N-1))

# defining the main diagonal
for i in range(0, N-1):
    A[i, i] = 2*(1 + r)
    B[i, i] = 2*(1 + r)
    
    # defining the other two diagonals
for i in range(0, N-2):
    A[i+1, i] = -r
    A[i, i+1] = -r
    B[i+1, i] = r
    B[i, i+1] = r
    
Ainv = np.linalg.inv(A)

plt.figure(figsize=(12, 4));
plt.subplot(121)
plt.imshow(A, interpolation='none');

ticks1 = np.arange(N-1)
ticks2 = np.arange(1, N-0.9, 1)
plt.xticks(ticks1, ticks2);
plt.yticks(ticks1, ticks2);

clb = plt.colorbar();
clb.set_label('Matrix elements values');
#clbs.set_clim((-1, 1))
plt.title('Matrix, $A$ r=%s'%(np.round(r, 3)), fontsize=12)

plt.subplot(122)
plt.imshow(B, interpolation='none');

ticks1 = np.arange(N-1)
ticks2 = np.arange(1, N-0.9, 1)
plt.xticks(ticks1, ticks2);
plt.yticks(ticks1, ticks2);

clb = plt.colorbar();
clb.set_label('Matrix elements values');
#clbs.set_clim((-1, 1))
plt.title('Matrix, $B$ r=%s'%(np.round(r, 3)), fontsize=12)


# Calculation step
fig = plt.figure(figsize=(12, 6))
plt.subplot(121)

for j in range(0, time_steps+1):
    b[0] = r*w[0, j-1] + r*w[0, j]
    b[N-2] = r*w[N, j-1] + r*w[N, j]
    v = np.dot(B, w[1:(N), j-1])
    w[1:(N), j] = np.dot(Ainv, v+b)
    
    plt.plot(x, w[:, j], 'o:', label='t[%s]=%s'%(j, time[j]))

plt.xlabel('x')
plt.ylabel('w')
#plt.legend(loc='bottom', bbox_to_anchor=(0.5, -0.1))
plt.legend(bbox_to_anchor=(-.4, 1), loc=2, borderaxespad=0.)

plt.subplot(122)
plt.imshow(w.transpose())
plt.xticks(np.arange(len(x)), x)
plt.yticks(np.arange(len(time)), time)
plt.xlabel('x')
plt.ylabel('time')
clb = plt.colorbar();
clb.set_label('Temperature (w)');
#clbs.set_clim((-1, 1))
plt.title('Numerical Solution of the Heat Equation r=%s'%(np.round(r, 3)), fontsize=12, y=1.08)

plt.tight_layout()
plt.show()