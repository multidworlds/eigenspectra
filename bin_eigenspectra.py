import numpy as np

def bin_eigenspectra(spectra, kgroups):
    '''
    Input an array of spectra and weights to produce eigenspectra
    spectra: array of Fp/Fs (wavelengths x lat x lon)
    kgroups: array of group indices (lat x lon)

    output a list of eigenspectra
    '''

    # Calculate the number of groups
    # assuming kgroups contains integers from 0 to ngroups
    ngroups = np.max(kgroups)+1
    
    nbins = spectra.shape[0] # number of wavelength bins
    # Flatten over latxlon
    spectra = spectra.reshape(nbins,-1)
    kgroups = kgroups.reshape(-1)

    eigenspectra = []
    for g in range(ngroups):
        ingroup = (kgroups == g).astype(int)
        eigenspec = np.sum(spectra*ingroup, axis=1)/np.sum(ingroup) 
        # eigenspec is the mean of spectra in group
        eigenspectra.append(eigenspec)
    
    return eigenspectra
