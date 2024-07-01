import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import time

from main1_functions import *

dir_path = os.path.dirname(os.path.realpath(__file__))+'\\Animations\\'
print(dir_path)
currenttime = time.time()
file_path = dir_path+f'animation_{currenttime}.mp4'

# Initialization _______________________________________________________________

# prvi del _____________________________________________________________
k = 1.
w = k**0.5

w = 0.2
k = w**2.
lamb = 10.
alpha = k**(1/4)

a = -40.
b = 40.
t0 = 0.
tnih = 2*np.pi / w
tf = 314.
tf = 5.*tnih

N = 5000
M = 5000

dx = (b - a)/(N-1)
dt = (tf - t0)/(M-1)

x = np.linspace(a, b, N)
t = np.linspace(t0, tf, M)
#t = np.arange(t0, tf, dt)


# drugi del ____________________________________________________________
sigma0 = 1./20.
k0 = 50.*np.pi
lamb2 = 0.25

a2 = -0.5
b2 = 1.5
t02 = 0.
tf2 = 1.
tf2 = .005

dx2 = (b2 - a2)/(N-1)
dt2 = 2.*dx**2
#x2 = np.arange(a2, b2+dx2, dx2)
#t2 = np.arange(t02, tf2, dt)

x2 = np.linspace(a2, b2, N)
t2 = np.linspace(t02, tf2, M)


# Calculations _________________________________________________________________

# prvi del
phi1_anal = analytic_koherent2(x, t)
zp1 = valovna_prvi(x)
phi1 = C_N(x, t, zp1)

napaka1 = np.abs(phi1 - phi1_anal)
napaka1sq = np.abs(np.abs(phi1**2) - np.abs(phi1_anal**2))

# drugi del
phi2_anal = analytic_empty2(x2, t2)
zp2 = valovna_drugi(x2)
phi2 = C_N(x2, t2, zp2)

napaka2 = np.abs(phi2 - phi2_anal)
napaka2sq = np.abs(np.abs(phi2**2) - np.abs(phi2_anal**2))

# Plotting _____________________________________________________________________
from matplotlib.animation import FuncAnimation

# Function to create animations
def create_animation(x, t, numerical_solution, analytical_solution, save_path):
    fig, ax = plt.subplots(figsize=(15, 6))
    if np.sum(numerical_solution[0]) == np.sum(phi1[0]):
        ax.plot(x, Vpot(x), 'k--', lw=0.7, alpha=0.7, label='Harmonic potential')
    line_anal, = ax.plot([], [], 'r', lw=0.8, alpha=0.9, label='Analytical Solution')
    line_num, = ax.plot([], [], 'b', lw=0.8, alpha=0.9, label='Numerical Solution')
    ax.legend()
    
    if np.sum(numerical_solution[0]) == np.sum(phi1[0]):
        ax.set_xlim((a-dx, b+dx))
        ax.set_ylim((0, 1.1*np.max(np.abs(phi1[0]**2))))
    else:
        ax.set_xlim((a2-dx2, b2+dx2))
        ax.set_ylim((-1.1*np.max(np.abs(phi2[0]**2)), 1.1*np.max(np.abs(phi2[0]**2))))


    def init():
        line_anal.set_data([], [])
        line_num.set_data([], [])
        return line_num, line_anal

    def update1(frame):
        print(frame)
        line_anal.set_data(x, analytical_solution[frame]**2)
        line_num.set_data(x, numerical_solution[frame]**2)
        ax.set_title(f'Time = {t[frame]:.4f} s')
        return line_num, line_anal
    
    def update2(frame):
        print(frame)
        line_anal.set_data(x, np.real(analytical_solution[frame]))
        line_num.set_data(x, np.real(numerical_solution[frame]))
        ax.set_title(f'Time = {t[frame]:.4f} s')
        return line_num, line_anal

    if np.sum(numerical_solution[0]) == np.sum(phi1[0]):
        ani = FuncAnimation(fig, update1, frames=len(t), init_func=init, blit=True)
    else:
        ani = FuncAnimation(fig, update2, frames=len(t), init_func=init, blit=True)
    ani.save(save_path, writer='ffmpeg', fps=30, dpi=150)

# Assuming you have already run the calculation for phi1
# You can replace phi1_numerical, phi1_analytical with your actual numerical and analytical solutions
#create_animation(x, t, phi1, phi1_anal, file_path)
create_animation(x2, t2, phi2, phi2_anal, file_path)
