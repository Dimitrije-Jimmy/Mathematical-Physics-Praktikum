import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation


# Seed value
# Apparently you may use different seed values at each stage
seed_value= 10001

# importing datetime module for now()
import datetime
 
# using now() to get current time
current_time = datetime.datetime.now()
print(current_time)
seed_value = current_time.second

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)



# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

#sampling
N = 5000    # število korakov


mu = 2.5
nu = 3
vkonst = 5.
tkonst = 1


def linear(x, k, n):
    return k*x + n

def distance(x, y):
    return (x**2 + y**2)**0.5




def walk2(N, mu=3, nu=3):
    global vkonst, tkonst

    x_sez = np.zeros(N)
    y_sez = np.zeros(N)
    time1 = np.zeros(N)     # the time associated with walk
    time2 = np.zeros(N)     # the time associated trapping

    theta = np.random.random(N)*2*np.pi
    #pareto distribution, pazi na (a+1) potenco!
    #a, m = 25., 2.  # shape and mode
    m = 2
    l = (np.random.pareto(mu-1, N) + 1) * m

    
    time2 = (np.random.pareto(nu-1, N) + 1) * m

    for i in range(1, N):
        x_sez[i] = x_sez[i-1] + l[i]*np.cos(theta[i])
        y_sez[i] = y_sez[i-1] + l[i]*np.sin(theta[i])
        time1[i] = time1[i-1] + l[i]/vkonst
        
        
    razdalja1 = distance(x_sez, y_sez)

    return x_sez, y_sez, time1, time2, theta, l, razdalja1


x_coord, y_coord, time1, time2, _, _, razdalja1 = walk2(N)


# Plotting _____________________________________________________________________
plt.figure(figsize=(15, 10))


plt.subplot(2, 2, 1)

plt.plot(x_coord[:10], y_coord[:10], 'b-', alpha=0.7, linewidth=0.2)
plt.scatter(x_coord[:10], y_coord[:10], c=y_coord[:10], cmap='viridis_r')

plt.plot(x_coord[0], y_coord[0], 'k.', label=r'Start $t=0$')
plt.plot(x_coord[9], y_coord[9], 'r.', label=r'End $t=10$')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(r'$N = 10$')


plt.subplot(2, 2, 2)

plt.plot(x_coord[:100], y_coord[:100], 'b-', alpha=0.7, linewidth=0.2)
plt.scatter(x_coord[:100], y_coord[:100], c=y_coord[:100], cmap='viridis_r')

plt.plot(x_coord[0], y_coord[0], 'k.', label=r'Start $t=0$')
plt.plot(x_coord[99], y_coord[99], 'r.', label=r'End $t=10^2$')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(r'$N = 10^2$')


plt.subplot(2, 2, 3)

plt.plot(x_coord[:1000], y_coord[:1000], 'b-', alpha=0.7, linewidth=0.2)
plt.scatter(x_coord[:1000], y_coord[:1000], c=y_coord[:1000], s=2, cmap='viridis_r')

plt.plot(x_coord[0], y_coord[0], 'k.', label=r'Start $t=0$')
plt.plot(x_coord[999], y_coord[999], 'r.', label=r'End $t=10^3$')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(r'$N = 10^3$')


plt.subplot(2, 2, 4)

plt.plot(x_coord, y_coord, 'b-', alpha=0.7, linewidth=0.2)
plt.scatter(x_coord, y_coord, c=y_coord, s=1, cmap='viridis_r')

plt.plot(x_coord[0], y_coord[0], 'k.', label=r'Start $t=0$')
plt.plot(x_coord[-1], y_coord[-1], 'r.', label=r'End $t=10^4$')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(r'$N = 10^4$')


plt.tight_layout()
plt.show()
plt.clf()
plt.close()


# Animation ____________________________________________________________________

from matplotlib.animation import FuncAnimation
from matplotlib import animation


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
"""
writergif = animation.PillowWriter(fps=30)
anim.save(directory+'\\'+'animation.gif', writer=writergif)
"""

anim.save(directory+'\\'+'animation_final.mp4', writer='ffmpeg')

plt.show()
