import numpy as np
import matplotlib.pyplot as plt

def analytic_finite(x, k, A=5.0):
    result = np.empty_like(x)

    for i, val in enumerate(x):
        if np.abs(val) <= 1:
            if k % 2 == 0:
                result[i] = np.sin(k / (2 * A) * np.pi * val)
            else:
                result[i] = np.cos(k / (2 * A) * np.pi * val)
        else:
            result[i] = np.exp(-A * np.abs(val))

    return result

# Generate data
x = np.linspace(-2, 2, 400)
k_values = [1, 2, 3]

# Plot multiple curves and annotate each one
for k in k_values:
    y = analytic_finite(x, k)
    plt.plot(x, y, label=f'k={k}')

    # Find the maximum y-value in the plot
    max_y = max(y)
    
    # Find the corresponding x-value for the maximum y
    max_x = x[np.argmax(y)]

    # Annotate the point
    plt.annotate(f'k={k}', xy=(max_x, max_y), xytext=(max_x, max_y + 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))

# Add labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.title('Annotated Curves')
plt.legend()

# Show the plot
plt.show()
