from eigensource import plot_utils
import numpy as np
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
import pdb

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
    plt.close(fig)


def plot_hue_maps(degree=3,model='hotspot'):
    dataDir = 'data/sph_harmonic_coefficients_full_samples/{}/'.format(model)
    ngroups = 2
    waves, lats, lons = plot_utils.get_map_and_plot(waveInd=8,degree=degree,dataDir=dataDir,isspider=False)
    
    npzLightcurve = np.load('data/input_lightcurves/hotspot.npz')
    time = npzLightcurve['time']
    extent=(np.max(time)-np.min(time))/2.21857567+180./360. #phase coverage of the eclipse observations

    allOutput = plot_utils.find_groups(dataDir,ngroups=2,degree=degree,isspider=False,extent=extent)
    eigenspectra_draws, kgroup_draws,uber_eigenlist, maps = allOutput
    

    # npzResults = np.load('data/sph_harmonic_coefficients_full_samples/hotspot/spherearray_deg_3.npz')
    # resultDict = npzResults['arr_0'].tolist()
    # waves = resultDict['wavelength (um)']
    
    ## Plot the Hue maps
    kgroups = plot_utils.show_spectra_of_groups(eigenspectra_draws,kgroup_draws,uber_eigenlist,waves)
    plot_utils.do_hue_maps(extent,maps,lons,lats,kgroups,ngroups,hueType='group')
    plt.savefig('plots/paper_figures/HUEgroup_LUMflux_{}_deg_{}.pdf'.format(model,degree), dpi=300, bbox_inches='tight')

    plot_utils.do_hue_maps(extent,maps,lons,lats,kgroups,ngroups,hueType='flux')
    plt.savefig('plots/paper_figures/HUEflux_LUMstdev_{}_deg_{}.pdf'.format(model,degree), dpi=300, bbox_inches='tight')
    
    

if __name__ == "__main__":
    plot_hotspot_derived_spectra()
