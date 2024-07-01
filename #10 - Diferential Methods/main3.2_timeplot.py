import numpy as np
import matplotlib.pyplot as plt
import os


dir_path = os.path.dirname(os.path.realpath(__file__))+'\\Results\\'
print(dir_path)


fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))                  # PLOT 1

# Function to read results from file and plot
def plot_results(file_path):
    # Load results from file using numpy
    results = np.load(file_path)

    N = results['N']
    M = results['M']
    napaka1 = results['napaka1']
    napaka1sq = results['napaka1sq']
    elapsed_time_phi1 = results['elapsed_time_phi1']
    napaka2 = results['napaka2']
    napaka2sq = results['napaka2sq']
    elapsed_time_phi2 = results['elapsed_time_phi2']


    # Error ____________________________________________________________
    #plt.subplot(1, 2, 1)
    #ax1.plot(N, np.mean(napaka1sq), 'b--', lw=0.8, alpha=0.7, label='prvi paket')
    #ax1.plot(N, np.mean(napaka2sq), 'r--', lw=0.8, alpha=0.7, label='drugi paket')
    ax1.scatter(N, np.mean(napaka1sq), color='b', marker='o')
    ax1.scatter(N, np.mean(napaka2sq), color='r', marker='o')
    #ax1.scatter(N, np.max(napaka1sq), color='b', marker='o')
    #ax1.scatter(N, np.max(napaka2sq), color='r', marker='o')


    # Execution time ___________________________________________________
    #plt.subplot(1, 2, 2)
    #ax2.plot(N, np.mean(elapsed_time_phi1), 'b--', lw=0.8, alpha=0.7, label='prvi paket')
    #ax2.plot(N, np.mean(elapsed_time_phi2), 'r--', lw=0.8, alpha=0.7, label='drugi paket')
    ax2.scatter(N, np.mean(elapsed_time_phi1), color='b', marker='o')
    ax2.scatter(N, np.mean(elapsed_time_phi2), color='r', marker='o')
    #ax2.scatter(N, np.max(elapsed_time_phi1), color='b', marker='o')
    #ax2.scatter(N, np.max(elapsed_time_phi2), color='r', marker='o')
    
    
    # Both _____________________________________________________________
    #ax3.plot(np.mean(elapsed_time_phi1), np.mean(napaka1sq), 'b--', lw=0.8, alpha=0.7, label='prvi paket')
    #ax3.plot(np.mean(elapsed_time_phi2), np.mean(napaka2sq), 'r--', lw=0.8, alpha=0.7, label='drugi paket')
    ax3.scatter(np.mean(elapsed_time_phi1), np.mean(napaka1sq), color='b', marker='o')
    ax3.scatter(np.mean(elapsed_time_phi2), np.mean(napaka2sq), color='r', marker='o')
    #ax3.scatter(np.max(elapsed_time_phi1), np.max(napaka1sq), color='b', marker='o')
    #ax3.scatter(np.max(elapsed_time_phi2), np.max(napaka2sq), color='r', marker='o')



# Plotting results for each file
#L_values = np.linspace(100, 2000, 50, dtype=int)
L_values = np.arange(100, 2100, 40, dtype=int)

for n in L_values:
    print(n)
    file_path = dir_path+f"results_N_{n}_M_{n}.npz"
    plot_results(file_path)
    
    
#ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.set_title('Absolute Error vs N and M')
ax1.set_xlabel('N = M')
ax1.set_ylabel('Mean Absolute Error Squared')
#ax1.legend()    

#ax2.set_xscale('log')
#ax2.set_yscale('log')
ax2.set_title('Execution Time vs N and M')
ax2.set_xlabel('N = M')
ax2.set_ylabel('Mean Execution Time (s)')
#ax2.legend()

#ax3.set_xscale('log')
#ax3.set_yscale('log')
ax3.set_title('Absolute Error vs Execution Time')
ax3.set_xlabel('Mean Execution Time (s)')
ax3.set_ylabel('Mean Absolute Error Squared')
#ax3.legend()

plt.tight_layout()
plt.show()
