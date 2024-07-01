import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statistics import mean, mode

def plot_grades_statistics(grades, exam):
    # Calculate mean, median, and mode
    grades_mean = mean(grades)
    grades_median = np.median(grades)
    grades_mode = mode(grades)

    fig = plt.figure()
    # Plot the histogram
    plt.hist(grades, bins=10, alpha=0.7, color='blue', edgecolor='black')

    # Add a vertical line at the mean, median, and mode
    plt.axvline(grades_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {grades_mean:.2f}')
    plt.axvline(grades_median, color='green', linestyle='dashed', linewidth=2, label=f'Median: {grades_median:.2f}')
    plt.axvline(grades_mode, color='orange', linestyle='dashed', linewidth=2, label=f'Mode: {grades_mode}')

    # Fit a Gaussian distribution to the data
    mu, std = norm.fit(grades)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    #plt.plot(x, p * len(grades), 'k', linewidth=2, label=f'Gaussian Fit\nMean: {mu:.2f}, Std: {std:.2f}')

    # Display standard deviation value
    plt.text(0.95, 0.85, f'Std: {std:.2f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', alpha=0.1))

    # Customize the plot
    plt.title(f'Class Grades Distribution {exam}')
    plt.xlabel('Grades')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)

    # Show the plot
    #plt.show()

# Example usage:
#grades = [75, 80, 85, 90, 95, 80, 85, 88, 92, 94, 88, 75, 90, 92, 95, 78, 88, 90]
#plot_grades_statistics(grades)


izpitA = [
    2.875,
    2.5,
    2.375,
    2.625,
    2.5,
    2.5,
    2.375,
    2.75,
    2.125,
    2.5,
    2.0,
    2.125,
    2.0,
    2.125,
    1.75,
    2.125,
    1.75,
    1.75,
    2.125,
    2.0,
    2.25,
    2.75,
    2.125,
    2.25,
    1.625,
    2.0,
    2.0,
    1.75,
    2.125,
    1.75,
    1.375,
    1.875,
    1.875,
    2.125,
    1.875,
    1.5,
    1.25,
    1.75,
    1.375,
    1.375,
    1.125,

    1.5,
    1.25,
    1.125,
    1.25,
    1.25,
    1.125,
    1.25,
    1.25,
    1.375,
    1.375,
    1.0,
    1.0,
    0.75,
    0.75,
    0.75,
    0.625,
    0.375
]

izpitB = [
    2.625,
    3.0,
    3.0,
    2.75,
    2.875,
    2.875,
    2.625,
    2.25,
    2.625,
    2.0,
    2.5,
    2.375,
    2.375,
    2.25,
    2.625,
    2.25,
    2.375,
    2.25,
    1.875,
    2.0,
    1.625,
    1.125,
    1.75,
    1.5,
    2.125,
    1.5,
    1.375,
    1.625,
    1.25,
    1.625,
    1.75,
    1.25,
    1.25,
    1.0,
    1.0,
    1.375,
    1.5,
    1.0,
    1.375,
    1.375,
    1.625,
    0.75,
    0.875,
    1.0,
    0.875,
    0.625,
    0.625,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
]

plot_grades_statistics(izpitA, 'A')
plot_grades_statistics(izpitB, 'B')


plt.show()