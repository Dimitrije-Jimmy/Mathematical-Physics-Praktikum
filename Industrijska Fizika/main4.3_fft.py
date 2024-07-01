import numpy as np
import matplotlib.pyplot as plt


def read_sinogram(file_path):
    return np.loadtxt(file_path).T

def filtersinc(PR):
    # Parameter "a" that varies the filter magnitude response
    a = 1
    
    Length, Count = PR.shape
    w = np.linspace(-np.pi, np.pi - 2*np.pi/Length, Length)

    rn1 = np.abs(2/a * np.sin(a * w/2))
    rn2 = np.sin(a * w/2)
    rd = (a * w)/2
    r = rn1 * (rn2/rd)**2

    f = np.fft.fftshift(r)
    
    g = np.zeros_like(PR, dtype=float)
    for i in range(Count):
        IMG = np.fft.fft(PR[:, i])
        fimg = IMG * f
        g[:, i] = np.fft.ifft(fimg).real

    return g


def backproject3(filtPR, THETA):
    # figure out how big our picture is going to be
    n = filtPR.shape[0]
    sideSize = n

    # filter the projections
    #filtPR = projfilter(PR)
    # filtPR = filterplus(PR)
    # filtPR = PR

    # convert THETA to radians
    #th = (np.pi / 180) * THETA
    th = THETA

    # set up the image
    m = len(THETA)
    BPI = np.zeros((sideSize, sideSize))

    # find the middle index of the projections
    midindex = (n + 1) / 2

    # set up x and y matrices
    x = np.arange(1, sideSize + 1)
    y = np.arange(1, sideSize + 1)
    X, Y = np.meshgrid(x, y)
    xpr = X - (sideSize + 1) / 2
    ypr = Y - (sideSize + 1) / 2

    # loop over each projection
    for i in range(m):
        print('On angle', THETA[i])

        # figure out which projections to add to which spots
        filtIndex = np.round(midindex + xpr * np.sin(th[i]) - ypr * np.cos(th[i])).astype(int)

        # if we are "in bounds" then add the point
        BPIa = np.zeros((sideSize, sideSize))
        spota = np.where((filtIndex > 0) & (filtIndex <= n))
        newfiltIndex = filtIndex[spota]-1
        BPIa[spota] = filtPR[newfiltIndex, i]
        BPI = BPI + BPIa

    BPI = BPI / m

    return BPI



def main(sinogram_file):
    # Read sinogram data
    sinogram = read_sinogram(sinogram_file)
    
    # Define the angles for each projection
    theta = np.linspace(0, np.pi, sinogram.shape[0], endpoint=False)
    
    # Filter the sinogram
    #filtered_sinogram = filtersinc(sinogram)
    filtered_sinogram = sinogram
    
    # Back projection
    reconstruction = backproject3(filtered_sinogram, theta)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(sinogram, cmap='gray', extent=[0, 180, -1, 1], aspect='auto', origin='lower')
    axes[0].set_xlabel(r'$\phi$')
    axes[0].set_ylabel(r'$s$')
    axes[0].set_title('Sinogram')
    
    axes[1].imshow(filtered_sinogram, cmap='gray', extent=[0, 180, -1, 1], aspect='auto', origin='lower')
    axes[1].set_xlabel(r'$\phi$')
    axes[1].set_ylabel(r'$s$')
    axes[1].set_title('Filtriran sinogram')

    axes[2].imshow(reconstruction, cmap='gray', extent=[-1, 1, -1, 1], aspect='auto', origin='lower')
    axes[2].set_xlabel(r'$x$')
    axes[2].set_ylabel(r'$y$')
    axes[2].set_title('Rekonstruirana slika')

    plt.tight_layout()
    #plt.show()
    
    return sinogram, filtered_sinogram, reconstruction

if __name__ == "__main__":
    
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))+'\\'
    print(dir_path)

    file_phantom = dir_path+'phantom.dat'
    file_sg1 = dir_path+'sg1.dat'
    file_sg2 = dir_path+'sg2.dat'
    
    main(file_phantom)
    main(file_sg1)
    main(file_sg2)
    
    plt.show()
