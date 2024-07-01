import numpy as np
import matplotlib.pyplot as plt
import csv

import os
dir_path = os.path.dirname(os.path.realpath(__file__))+'\\Images\\Results\\'
print(dir_path)

# Functions ____________________________________________________________________

def read_results_from_csv(filename):
    results = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            results.append((
                int(row['M']),
                int(row['N']),
                float(row['C']),
                np.array(row['a']),
                float(row['err_C']),
                float(row['execution_time'])
            ))
    return results


def plot_error_vs_M(results):
    M_values = np.unique([result[0] for result in results])
    err_C_values = np.array([result[4] for result in results])

    plt.plot(M_values, err_C_values, marker='o')
    plt.xlabel('M')
    plt.ylabel('Absolute Error of C in relation to C100')
    plt.title('Error vs M')
    plt.show()

def plot_error_vs_N(results):
    M_values = np.unique([result[0] for result in results])
    N_values = np.unique([result[1] for result in results])
    err_C_values = np.array([result[4] for result in results])
    
    x_sez = [i+j for j in N_values for i in M_values]

    plt.plot(x_sez, err_C_values, marker='o')
    plt.xlabel('N')
    plt.ylabel('Absolute Error of C in relation to C100')
    plt.title('Error vs N')
    plt.show()

def plot_execution_time_vs_M(results):
    M_values = np.unique([result[0] for result in results])
    execution_time_values = np.array([result[5] for result in results])

    plt.plot(M_values, execution_time_values, marker='o')
    plt.xlabel('M')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs M')
    plt.show()

def plot_execution_time_vs_N(results):
    N_values = np.unique([result[1] for result in results])
    execution_time_values = np.array([result[5] for result in results])

    plt.plot(N_values, execution_time_values, marker='o')
    plt.xlabel('N')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs N')
    plt.show()


def plot_matrix_imshow(results):
    M_values = np.unique([result[0] for result in results])
    N_values = np.unique([result[1] for result in results])
    err_C_matrix = np.array([[result[4] for result in results if result[0] == M and result[1] == N][0] for M in M_values for N in N_values])

    plt.imshow(err_C_matrix, cmap='viridis', origin='lower', extent=[min(N_values), max(N_values), min(M_values), max(M_values)])
    plt.colorbar(label='Absolute Error of C in relation to C100')
    plt.xlabel('N')
    plt.ylabel('M')
    plt.title('Error Matrix - Imshow')
    plt.show()


def plot_matrix_contourf(results):
    M_values = np.unique([result[0] for result in results])
    N_values = np.unique([result[1] for result in results])
    err_C_matrix = np.array([[result[4] for result in results if result[0] == M and result[1] == N][0] for M in M_values for N in N_values])

    plt.contourf(N_values, M_values, err_C_matrix, cmap='viridis')
    plt.colorbar(label='Absolute Error of C in relation to C100')
    plt.xlabel('N')
    plt.ylabel('M')
    plt.title('Error Matrix - Contourf')
    plt.show()
    
# Execution and plotting _______________________________________________________

# Read results from CSV
results_read = read_results_from_csv(dir_path+'output.csv')

# Generate plots
plot_error_vs_N(results_read)
#plot_execution_time_vs_N(results_read)
#plot_matrix_imshow(results_read)
#plot_matrix_contourf(results_read)

