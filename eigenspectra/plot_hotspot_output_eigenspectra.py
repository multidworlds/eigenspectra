from eigensource import plot_utils
import numpy as np
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
import pdb
import eigenmaps

def plot_hotspot_derived_spectra():
    #dataDir = 'data/sph_harmonic_coefficients_full_samples/hotspot/'
    #dataDir = 'data/sph_harmonic_coefficients_full_samples/hotspot_3curves/'
    dataDir = 'data/sph_harmonic_coefficients_full_samples/finalgood/hotspot64/'
    allOutput = plot_utils.find_groups(dataDir,ngroups=2,degree=3,isspider=False)

    eigenspectra_draws, kgroup_draws,uber_eigenlist, maps = allOutput

    npzResults = np.load('data/sph_harmonic_coefficients_full_samples/hotspot/spherearray_deg_3.npz', allow_pickle=True, encoding="latin1")
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


def plot_group_histos(ngroups=2,degree=3):
    ## get te groups
    model = 'hotspot64'

    dataDir = 'data/sph_harmonic_coefficients_full_samples/finalgood/{}/'.format(model)

    npzLightcurve = np.load('data/input_lightcurves/hotspot.npz')
    time = npzLightcurve['time']
    extent=(np.max(time)-np.min(time))/2.21857567+180./360. #phase coverage of the eclipse observations

    allOutput = plot_utils.find_groups(dataDir,ngroups=ngroups,degree=degree,isspider=False,extent=extent)

    eigenspectra_draws, kgroup_draws,uber_eigenlist, maps = allOutput

    ## get the waves, longitudes and latitudes
    waves, lats, lons = plot_utils.get_map_and_plot(waveInd=8,degree=degree,dataDir=dataDir,isspider=False)



    kgroups = plot_utils.show_spectra_of_groups(eigenspectra_draws,kgroup_draws,uber_eigenlist,waves)
    ## plot the histos
    saveName = 'plots/paper_figures/group_histos_{}_degree_{}_ngroup_{}.pdf'.format(model,degree, ngroups)
    xLons= np.array([-50,-10,40,59]) * np.pi/180.
    xLats= np.array([40,5, 0,-59]) * np.pi/180.

    eigenmaps.show_group_histos(kgroups,lons,lats,kgroup_draws,
                                xLons=xLons,xLats=xLats,
                                saveName=saveName,figsize=(3,2))

def plot_all_group_histos():
    for ngroup in [2,3]:
    #    for degree in [2,3]:
        degree=3
        plot_group_histos(ngroup,degree)

def plot_hue_maps(degree=3,model='hotspot64'):
    dataDir = 'data/sph_harmonic_coefficients_full_samples/finalgood/{}/'.format(model)
    ngroups = 2
    waves, lats, lons = plot_utils.get_map_and_plot(waveInd=8,degree=degree,dataDir=dataDir,isspider=False)

    npzLightcurve = np.load('data/input_lightcurves/hotspot.npz', allow_pickle=True, encoding="latin1")
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
