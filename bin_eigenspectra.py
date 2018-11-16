import numpy as np


def bin_eigenspectra(spectra, kgroups):
    '''
    Converts a grid of spectra into eigenspectra defined by kgroups.

    Parameters
    ----------
    spectra : array of Fp/Fs (axes: wavelengths x lat x lon)
    kgroups : array of group indices (ints) from 0 to k-1 (axes: lat x lon)

    Returns
    -------
    eigenspectra : list of k spectra, averaged over each group
    '''

    # Calculate the number of groups
    # assuming kgroups contains integers from 0 to ngroups
    ngroups = np.max(kgroups)+1

    nbins = spectra.shape[0]  # number of wavelength bins
    spectra = spectra.reshape(nbins, -1)  # Flatten over (lat x lon)
    kgroups = kgroups.reshape(-1)

    eigenspectra = []
    for g in range(ngroups):
        ingroup = (kgroups == g).astype(int)
        eigenspec = np.sum(spectra*ingroup, axis=1)/np.sum(ingroup)
        # eigenspec is the mean of spectra in group
        eigenspectra.append(eigenspec)

    return eigenspectra

