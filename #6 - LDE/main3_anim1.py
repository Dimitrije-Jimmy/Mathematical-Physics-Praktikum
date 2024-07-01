import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import psutil
import os

from diffeq import *


def f( x, t, k=0.1, T_out=-5.0 ):
    return -k*(x - T_out)

def analyt( t, x0, k=0.1, T_out=-5.0 ):
    return  T_out + np.exp(-k*t)*(x0 - T_out)


a, b = ( 0.0, 31.0 )

x0 = 21.0
t = np.linspace( a, b, 10 )
x_analyt = analyt(t, x0)

h_sez = [2.0, 1.0, 0.5, 0.1, 0.001, 1e-4]



# Set up parameters
t = np.linspace(0, 10, 1000)
step_sizes = [1.0, 0.5, 0.2, 0.1]  # Add more step sizes as needed

# Initialize lists to store results for each step size and method
results_list = []

methods = [euler, heun, rk45]

for method in methods:
    results_for_method = []
    for h in step_sizes:
        # Perform integration and store the result in the list
        n = int((b-a)/np.abs(h))
        t = np.linspace( a, b, n )
        numerical_solution = method(f, x0, t)
        results_for_method.append(numerical_solution)

    results_list.append(results_for_method)

# Initialize the figure and axis
fig, ax = plt.subplots()

scatter = ax.scatter([], [], c='b', cmap='viridis_r', marker='o', edgecolor='k', s=50)
line, = ax.plot([], [], 'b-', lw=0.2, alpha=0.7)  # Line connecting the dots


# Initialize text for point number display
point_number_text = ax.text(0.85, 0.9, '', transform=ax.transAxes)

# Set labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax.legend(loc='upper right')

# Initialize the limits for xlim and ylim
x_min, x_max = min(x_coord), max(x_coord)
y_min, y_max = min(y_coord), max(y_coord)
ax.set_xlim(x_min - 10, x_max + 10)
ax.set_ylim(y_min - 10, y_max + 10)


# Function to create a gradient effect in points
def create_color_gradient(x, y, num_points, cmap='viridis_r'):
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, num_points))
    return colors

# Function to update the plot in each frame of the animation
def update(frame):
    print(frame)

    x = x_coord[:frame]
    y = y_coord[:frame]
    
    line.set_data(x, y)
    scatter.set_offsets(np.column_stack((x, y)))
    
    # Calculate the gradient effect for the current frame
    num_frames = len(x_coord)
    colors = create_color_gradient(x, y, num_frames)

    # Plot the points with the gradient effect
    scatter.set_color(colors[:frame])

    # Update xlim and ylim
    if frame == 0:
        x_min, x_max = 0, 0
        y_min, y_max = 0, 0
    else:
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)
    ax.set_xlim(x_min - 10, x_max + 10)
    ax.set_ylim(y_min - 10, y_max + 10)

    # Update point number display
    point_number_text.set_text(f'Point {frame + 1}')


    return scatter, line, point_number_text


# Set a custom frame interval (e.g., 10 milliseconds for a faster animation)
frame_interval = 10

# Create an animation
num_frames = len(x_coord)
anim = FuncAnimation(fig, update, frames=num_frames, repeat=False, blit=True, interval=frame_interval)

directory = os.path.dirname(__file__)

anim.save(directory+'\\'+'animation_final.mp4', writer='ffmpeg')

plt.show()





"""
# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
plt.subplots_adjust(hspace=0.4)

def update(frame):

    # Access the pre-calculated results for the given step size and method
    print(results_list)
    step_size, numerical_solution = results_list[frame]
    x_analyt = analyt(t)

    # Plot the time evolution on the left subplot
    plt.subplot( 1, 2, 1 )
    plt.clear()
    plt.plot(t, numerical_solution, label=f'Numerical Solution (Method {frame + 1})')
    plt.plot(t, x_analyt, label='Analytical Solution', linestyle='dashed')
    plt.title('Time Evolution')
    plt.xlabel('Time')
    plt.ylabel('Temperature (T)')
    plt.legend()

    # Plot the absolute error on the right subplot
    absolute_error = np.abs(numerical_solution - x_analyt)
    plt.subplot( 1, 2, 2 )
    plt.clear()
    plt.plot(t, absolute_error, label=f'Absolute Error (Method {frame + 1})')
    plt.title('Absolute Error')
    plt.xlabel('Time')
    plt.ylabel('Absolute Error')
    plt.legend()

    

    print(f"Step Size: {step_size}")

ani = FuncAnimation(fig, update, frames=len(step_sizes), interval=1000, repeat=False)
plt.show()"""


# Plot the mesh plot of the result on the left
plt.plot_surface(t, np.linspace(0, 1, len(t)),
   np.tile(numerical_solution, (len(np.linspace(0, 1, len(t))), 1)),
   cmap='viridis', edgecolor='k')
plt.title('Mesh Plot - Numerical Solution')
plt.xlabel('Time')
plt.ylabel('Step Size')
plt.zlabel('Temperature (T)')

# Plot the mesh plot of the error on the right
absolute_error = np.abs(numerical_solution - x_analyt)
plt.subplot( 1, 2, 2 )
plt.plot_surface(t, np.linspace(0, 1, len(t)),
   np.tile(absolute_error, (len(np.linspace(0, 1, len(t))), 1)),
   cmap='viridis', edgecolor='k')
plt.title('Mesh Plot - Absolute Error')
plt.xlabel('Time')
plt.ylabel('Step Size')
plt.zlabel('Absolute Error')