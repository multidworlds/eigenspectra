from eigensource import plot_utils
import numpy as np
from astropy.io import fits, ascii

def plot_hotspot_derived_spectra():
    dataDir = 'data/sph_harmonic_coefficients_full_samples/hotspot/'
    allOutput = plot_utils.find_groups(dataDir,ngroups=2,degree=3,isspider=False)

    eigenspectra_draws, kgroup_draws,uber_eigenlist, maps = allOutput

    npzResults = np.load('data/sph_harmonic_coefficients_full_samples/hotspot/spherearray_deg_3.npz')
    resultDict = npzResults['arr_0'].tolist()
    waves = resultDict['wavelength (um)']
    
    fig = plot_utils.show_spectra_of_groups(eigenspectra_draws,kgroup_draws,uber_eigenlist,waves,
                                            saveName='kmeans_hotspot',degree=3,returnFig=True)
    
    ## plot the input spectra
    inputSpec = ascii.read('data/input_lightcurves/hotspot_2_spec.csv')
    axList = fig.axes
    ax = axList[0]
    
    ax.plot(inputSpec['wavelength (um)'],inputSpec['flux outside spot'],label='Input at Surrounding')
    ax.plot(inputSpec['wavelength (um)'],inputSpec['flux inside spot'],label='Input at Hotspot')
    
    ax.legend(fontsize=15)
    ax.set_title("Recovered Spectra")
    
    fig.savefig('plots/paper_figures/hot_spot_spectra_deg3_2groups_error_bars.pdf',
                bbox_inches='tight')

if __name__ == "__main__":
    plot_hotspot_derived_spectra()
