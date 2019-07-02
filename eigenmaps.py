# The input sph array will have the last two dimensions be (wavelengths,
# harmonic coefficients + 1). The first entry in each row is assumed to be
# the wavelength (in nm) corresponding to the spherical harmonic coefficients
# in that row. Note that the input array can have any dimensions prior to the
# last two.

# The returned array has dimensions (..., wavelengths, latitudes, longitudes),
# where the ellipses denote any extra dimensions from the input array.

import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as p
import pdb

def generate_maps(sph, N_lon, N_lat):
    '''
    Compute brightness map on a grid using spherical harmonic coefficients.

    Parameters
    ----------
    sph : array of spherical harmonic coefficients (axes: ..., wavelengths, SH coeffs)
    N_lon : int, number of gridpoints in longitude
    N_lat : int, number of gridpoints in latitude

    Returns
    -------
    wavelengths : array containing wavelength of each map
    lats : array of latitudes (radians)
    lons : array of longitudes (radians)
    fluxes : array of brightness maps at each wavelength, computed on grid
    '''

    wavelengths = sph[..., 0]
    harmonics = sph[..., 1:]
    degree = int(np.sqrt(np.shape(harmonics)[-1]))

    # The scipy spherical harmonic routine requires coordinates in polar form.
    las = np.linspace(0, np.pi, N_lat)
    los = np.linspace(0, 2*np.pi, N_lon)

    sph_l = np.concatenate([np.tile(l, l+1) for l in range(degree)])
    sph_m = np.concatenate([np.arange(l+1) for l in range(degree)])

    base_harmonics = sph_harm(np.tile(sph_m, (N_lon, N_lat, 1)).T,
                              np.tile(sph_l, (N_lon, N_lat, 1)).T,
                              *np.meshgrid(los, las))

    fluxes = np.sum([
                np.einsum('m...wvu,m->...wvu', np.array(
                        [np.einsum('...wx,xvu->...wvu',
                                   harmonics[..., l*(l+1)+np.array([m, -m])],
                                   np.array([base_harmonics[l*(l+1)//2+m].real,
                                             base_harmonics[l*(l+1)//2+m].imag
                                             ]))
                            for m in range(l+1)]),
                        [1] + l * [((-1)**l)*np.sqrt(2)])
                for l in range(degree)], axis=0)

    # Here we convert to (-pi, pi) in longitude, and (-pi/2, pi/2) in latitude,
    # and multiply by the factor that normalizes the harmonics.
    fluxes = 2*np.sqrt(np.pi) * \
        np.flip(np.roll(fluxes, N_lon//2, axis=-1), axis=-2)

    lons, lats = np.meshgrid(los-np.pi, las-np.pi/2)

    return wavelengths, lats, lons, fluxes

def show_group_histos(map_day,lons,lats,kgroup_draws,
                      xLons=[-0.5,-0.5,0.5, 0.5],
                      xLats=[-0.5, 0.5,0.5,-0.5],
                      global_map_units='Mean Group',
                      saveName=None):
    """
    Show histograms of the groups for specific regions of the map
    
    Parameters
    ----------
    map_day: 2D numpy array
        Dayside map of brightness or another quantity
        The map is the dayside only.
    lats: 2D numpy array
        Latitudes for the global_map grid in radians
    lons: 2D numpy array
        Longitudes for the global_map grid in radians
        It will assume that the dayside is lons[:,londim//4:-londim//4]
    kgroup_draws: 3D numpy array
        Kgroup draws from k Means Nsamples x Latitudes? x Longitudes?
    
    xLons: 4 element list or numpy array
        longitudes of points to show histograms in radians
    xLats: 4 element list or numpy array
        latitudes of points to show histograms in radians
    global_map_units: str
        Label for the global map units
    saveName: str or None
        Name of plot to save
    """
    
    
    londim = lons.shape[1]
    
    fig, ax = p.subplots()
    
    lon_day = lons[:,londim//4:-londim//4]
    lonMin = np.min(lon_day) * 180./np.pi
    lonMax = np.max(lon_day) * 180./np.pi
    plotData = ax.imshow(map_day, extent=[lonMin,lonMax,-90,90])
    cbar = fig.colorbar(plotData,ax=ax)
    cbar.set_label(global_map_units)
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')
    
    windowLocationsX = [-0.16,-0.16, 1.0, 1.0]
    windowLocationsY = [ 0.1,  0.6 , 0.6, 0.1]
    windowLabels = ['A','B','C','D']
    for ind in np.arange(len(xLons)):
        xLon, xLat = xLons[ind], xLats[ind]
        left, bottom, width, height = [windowLocationsX[ind], windowLocationsY[ind], 0.2, 0.2]
        ax2 = fig.add_axes([left, bottom, width, height])
        iLon, iLat = np.argmin(np.abs(lon_day[0,:] - xLon)), np.argmin(np.abs(lats[:,0] - xLat))
        ax.text(lon_day[0,iLon]* 180./np.pi,lats[iLat,0]* 180./np.pi,windowLabels[ind],
                color='red')
        
        ax2.set_title(windowLabels[ind])
        ax2.set_xlabel('Grp')
        
        ax2.hist(kgroup_draws[:,iLat,iLon])
        
    if saveName is not None:
        fig.savefig(saveName,bbox_inches='tight')
    #fig.suptitle('Retrieved group map, n={}, {:.2f}$\mu$m'.format(degree,waves[waveInd]))

    