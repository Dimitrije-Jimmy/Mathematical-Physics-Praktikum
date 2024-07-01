import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon


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


def main_fft(sinogram_file, filter=True):
    # Read sinogram data
    sinogram = read_sinogram(sinogram_file)
    
    # Define the angles for each projection
    theta = np.linspace(0, np.pi, sinogram.shape[0], endpoint=False)
    
    # Filter the sinogram
    if filter is True:
        filtered_sinogram = filtersinc(sinogram)
    else:
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

    reconstruction = np.fliplr(reconstruction.T)
    axes[2].imshow(reconstruction, cmap='gray', extent=[-1, 1, -1, 1], aspect='auto', origin='lower')
    axes[2].set_xlabel(r'$x$')
    axes[2].set_ylabel(r'$y$')
    axes[2].set_title('Rekonstruirana slika')

    # Set a main title for the entire figure
    fig.suptitle('Filtrirana povratna projekcija z uporabo FFT', fontsize=16)

    plt.tight_layout()
    #plt.show()
    
    return sinogram, filtered_sinogram, reconstruction


def main_iradon(sinogram_file):
    sinogram = read_sinogram(sinogram_file)

    theta =  np.linspace(0., 180., len(sinogram), endpoint=False)

    reconstruction = iradon(sinogram, theta=theta, circle=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
    axes[0].imshow(sinogram, cmap='gray', extent=[0, np.pi, -1, 1], aspect='auto', origin='lower')
    axes[0].set_xlabel(r'$\phi$')
    axes[0].set_ylabel(r'$s$')
    axes[0].set_title('Sinogram')

    #axes[1].imshow(np.fliplr(np.flipud(reconstruction)), cmap='gray', extent=[-1, 1, -1, 1], aspect='auto', origin='lower')
    #axes[1].imshow(np.flip(reconstruction), cmap='gray', extent=[-1, 1, -1, 1], aspect='auto', origin='lower')
    reconstruction = np.flipud(reconstruction)
    axes[1].imshow(reconstruction, cmap='gray', extent=[-1, 1, -1, 1], aspect='auto', origin='lower')
    axes[1].set_xlabel(r'$x$')
    axes[1].set_ylabel(r'$y$')
    axes[1].set_title('Rekonstruirana slika')
    
    # Set a main title for the entire figure
    fig.suptitle('Filtrirana povratna projekcija z uporabo Scikit-Image', fontsize=16)

    plt.tight_layout()
    #plt.show()
    
    return sinogram, reconstruction



if __name__ == "__main__":
    
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))+'\\'
    print(dir_path)

    file_phantom = dir_path+'phantom.dat'
    file_sg1 = dir_path+'sg1.dat'
    file_sg2 = dir_path+'sg2.dat'
    
    file = file_phantom
    file = file_sg1
    file = file_sg2
    
    # FFT brez filtra
    #sgram01, sgram_filt01, sgram_rec01 = main_fft(file_phantom, filter=False)
    #sgram02, sgram_filt02, sgram_rec02 = main_fft(file_sg1, filter=False)
    #sgram03, sgram_filt03, sgram_rec03 = main_fft(file_sg2, filter=False)
    
    # FFT z filtrom
    #sgram11, sgram_filt11, sgram_rec11 = main_fft(file_phantom)
    #sgram12, sgram_filt12, sgram_rec12 = main_fft(file_sg1)
    #sgram13, sgram_filt13, sgram_rec13 = main_fft(file_sg2)
    
    # Iradon
    #sgram21, sgram_rec21 = main_iradon(file_phantom)
    #sgram22, sgram_rec22 = main_iradon(file_sg1)
    #sgram23, sgram_rec23 = main_iradon(file_sg2)
    
    
    sgram0, sgram_filt0, sgram_rec0 = main_fft(file, filter=False)      # FFT brez filtra
    sgram1, sgram_filt1, sgram_rec1 = main_fft(file)                    # FFT z filtrom
    sgram2, sgram_rec2 = main_iradon(file)                              # Iradon
    
    # Some additional graphs
    #sgram_rec2 = sgram_rec23
    #sgram_rec0 = sgram_rec03
    #sgram_rec1 = sgram_rec13
    
    correct_reconstruction = sgram_rec2
    err0 = correct_reconstruction - sgram_rec0
    err1 = correct_reconstruction - sgram_rec1
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1.2, 1.2, 1.2]})
    
    im0 = axes[0].imshow(err0, cmap='gray', extent=[-1, 1, -1, 1], aspect='auto', origin='lower')
    axes[0].set_xlabel(r'$x$')
    axes[0].set_ylabel(r'$y$')
    axes[0].set_title('Nefiltriran sinogram')
    
    # Adding a colorbar between the first and second subplots
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im0, cax=cax)
    cbar.set_label('Velikost napake')
    
    im1 = axes[1].imshow(err1, cmap='gray', extent=[-1, 1, -1, 1], aspect='auto', origin='lower')
    axes[1].set_xlabel(r'$x$')
    axes[1].set_ylabel(r'$y$')
    axes[1].set_title('Filtriran sinogram')

    # Adding a colorbar between the second and third subplots
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im1, cax=cax)
    cbar.set_label('Velikost napake')

    im2 = axes[2].imshow(correct_reconstruction, cmap='gray', extent=[-1, 1, -1, 1], aspect='auto', origin='lower')
    axes[2].set_xlabel(r'$x$')
    axes[2].set_ylabel(r'$y$')
    axes[2].set_title('Rekonstruirana slika z funkcijo Iradon')
    
    # Adding a colorbar for the third subplot
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im2, cax=cax)
    cbar.set_label('Svetlost')

    # Set a main title for the entire figure
    fig.suptitle('Napaka povratnih projekcij', fontsize=16)

    plt.tight_layout()
    
    plt.show()
    plt.clf()
    plt.close()
    
    
