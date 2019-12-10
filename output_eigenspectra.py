from eigensource import plot_utils
import numpy as np

def plot_hotspot_derived_spectra():
    dataDir = 'data/sph_harmonic_coefficients_full_samples/hotspot/'
    allOutput = plot_utils.find_groups(dataDir,ngroups=2,degree=3,isspider=False)

    eigenspectra_draws, kgroup_draws,uber_eigenlist, maps = allOutput

    npzResults = np.load('data/sph_harmonic_coefficients_full_samples/hotspot/spherearray_deg_3.npz')
    resultDict = npzResults['arr_0'].tolist()
    waves = resultDict['wavelength (um)']

    plot_utils.show_spectra_of_groups(eigenspectra_draws,kgroup_draws,uber_eigenlist,waves,
                                      saveName='kmeans_hotspot',degree=3)
                                      
                                      
