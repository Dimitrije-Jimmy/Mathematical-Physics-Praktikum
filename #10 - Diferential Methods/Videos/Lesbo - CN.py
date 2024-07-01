# Grace P - Implementation of Crank-Nicolson in Python

import numpy as np
from plotly.subplots import make_subplots

# Crank - Nicolson not stable
nx = 50
nt = 2000

alpha = 1.0
d_x = 1.0
d_t = 0.5

x = np.arange(nx)*d_t
r = d_t*alpha/d_x**2

T = np.full(nx, 20.0)
T[0] = 50
T[-1] = 80

fig = make_subplots(rows=1, cols=1)
fig.add_scatter(x=x, y=T, name='initial temp')

for i in range(1, nt+1):
    T[1:-1] = T[1:-1] + r*(T[2:] - 2*T[1:-1] + T[:-2])

    if i % 100 == 0:
        fig.add_scatter(x=x, y=T, mode='lines', name=f'{i}')

fig.show()


# The actual video
# Crank-Nicolson stable for any value of r

import plotly.io as pio
from scipy.linalg import solve_banded

nx = 50
nt = 2000

alpha = 1.0
d_x = 0.5
d_t = 0.5

x = np.arange(nx)*d_t
r = d_t*alpha/d_x**2/2
# r = 1

T = np.full(nx, 20.0)
T[0] = 50
T[-1] = 80

A = np.zeros((3, nx-2))
A[0, 1:] = r
A[1, :] = 1 + 2.0*r
A[2, :-1] = -r
B = np.zeros(nx-2)
B = r*T[2:] + (1 - 2.0*r)*T[1:-1] + r*T[:-2]
B[0] = B[0] + r*T[0]
B[-1] = B[-2] + r*T[-1]


fig = make_subplots(rows=1, cols=1)
fig.add_scatter(x=x, y=T, name='initial profile')
for i in range(1, nt+1):
    T[1:-1] = solve_banded((1,1), A, B)
    B = r*T[2:] + (1 - 2.0*r)*T[1:-1] + r*T[:-2]
    B[0] = B[0] + r*T[0]
    B[-1] = B[-2] + r*T[-1]
    if i % 100 == 0:
        #fig.add_scatter(x=x, y=T, mode='lines', name=f'{i*d_t} s')
        fig.add_scatter(x=x, y=T, name=f'{i*d_t} s')

fig.show()