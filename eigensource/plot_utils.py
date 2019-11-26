import numpy as np
import matplotlib.pyplot as p

import eigencurves
import eigenmaps
import kmeans
import bin_eigenspectra
import os
import pdb
import spiderman as sp

import gen_lightcurves
import healpy as hp

import colorcet as cc
from colormap2d import generate_map2d
from matplotlib import colorbar, cm
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import run_higher_sph_harm

from importlib import import_module

def plot_setup():
    """
    Set some default plotting parameters
    """
    from matplotlib import rcParams
    rcParams["savefig.dpi"] = 200
    rcParams["figure.dpi"] = 100
    rcParams["font.size"] = 20
    rcParams["figure.figsize"] = [8, 5]
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Computer Modern Sans Serif"]
    rcParams["text.usetex"] = True

def show_orig_map(lam,spaxels,waveInd=0):#testNum=1):
    """
    Show the original map at a given wavelength

    Parameters
    -----------
    waveInd: int
        The wavelength index
    testNum: int
        The test Number (ie. lightcurve number)
    """
    #origData = np.load("data/maps/mystery_map{}.npz".format(testNum))
    lammin1 = 2.41; lammax1 = 3.98; dlam1 = 0.18
    #spaxels = origData["spaxels"]
    #lam = origData["wl"]
    lamlo, dlamlo = gen_lightcurves.construct_lam(lammin1, lammax1, dlam=dlam1)
    Nlamlo = len(lamlo)

    # Set HealPy pixel numbers
    Npix = spaxels.shape[0]

    # Define empty 2d array for spaxels
    spec2d = np.zeros((Npix, Nlamlo))

    # Loop over pixels filling with spectra
    for i in range(Npix):
        # Degrade the spectra to lower resolution
        spec2d[i,:] = gen_lightcurves.downbin_spec(spaxels[i, :], lam, lamlo, dlam = dlamlo)


    hp.mollview(spec2d[:,waveInd], title=r"%0.2f $\mu$m" %lamlo[waveInd])
    p.show()
    return spec2d

def retrieve_map_full_samples(degree=3,dataDir="data/sph_harmonic_coefficients_full_samples/hotspot/",isspider=True):
    tmp = np.load("{}spherearray_deg_{}.npz".format(dataDir,degree))
    outDictionary = tmp['arr_0'].tolist()

    londim = 100
    latdim = 100
    samples = outDictionary['spherical coefficients'] # output from eigencurves
    waves = outDictionary['wavelength (um)']
    bestsamples=outDictionary['best fit coefficients'] # best fit sample from eigencurves

    randomIndices = np.random.randint(0,len(samples),39)
    nRandom = len(randomIndices)

    fullMapArray = np.zeros([nRandom,len(waves),londim,latdim])
    bestMapArray = np.zeros([len(waves),londim,latdim])
    #SPIDERMAN stuff added by Megan
    if isspider:
        params0=sp.ModelParams(brightness_model='spherical')
        params0.nlayers=20
        params0.t0=-2.21857/2.
        params0.per=2.21857567
        params0.a_abs=0.0313
        params0.inc=85.71
        params0.ecc=0.0
        params0.w=90.
        params0.rp=0.155313
        params0.a=8.863
        params0.p_u1=0.
        params0.p_u2=0.
        params0.degree=degree
        params0.la0=0.
        params0.lo0=0.

    if not isspider:
        inputArr=np.zeros([len(waves),bestsamples.shape[0]+1])
        inputArr[:,0] = waves
        inputArr[:,1:] = bestsamples.transpose()
        wavelengths, lats, lons, maps = eigenmaps.generate_maps(inputArr,N_lon=londim, N_lat=latdim)
        bestMapArray=maps
    else:
        for i in np.arange(np.shape(waves)[0]):
            params0.sph=list(bestsamples[:,i])
            nla=latdim
            nlo=londim
            las = np.linspace(-np.pi/2,np.pi/2,nla)
            los = np.linspace(-np.pi,np.pi,nlo)
            fluxes = []
            for la in las:
                row = []
                for lo in los:
                    flux = sp.call_map_model(params0,la,lo)
                    row += [flux[0]]
                fluxes += [row]
            fluxes = np.array(fluxes)
            lons, lats = np.meshgrid(los,las)
            bestMapArray[i,:,:] = fluxes


    for drawInd, draw in enumerate(samples[randomIndices]):
        if not isspider:
            inputArr = np.zeros([len(waves),samples.shape[1]+1])
            inputArr[:,0] = waves
            inputArr[:,1:] = draw.transpose()

            wavelengths, lats, lons, maps = eigenmaps.generate_maps(inputArr,
                                                                N_lon=londim, N_lat=latdim)
            fullMapArray[drawInd,:,:,:] = maps
            #MEGAN ADDED STUFF
        else:
            for i in np.arange(np.shape(waves)[0]):
                params0.sph=list(samples[drawInd,:,i])
                nla=latdim
                nlo=londim
                las = np.linspace(-np.pi/2,np.pi/2,nla)
                los = np.linspace(-np.pi,np.pi,nlo)
                fluxes = []
                for la in las:
                    row = []
                    for lo in los:
                        flux = sp.call_map_model(params0,la,lo)
                        row += [flux[0]]
                    fluxes += [row]
                fluxes = np.array(fluxes)
                lons, lats = np.meshgrid(los,las)
            #print(np.min(lats),np.min(lons),np.min(las),np.min(los))
            #lats, lons, maps = testgenmaps.spmap(inputArr,londim, latdim)
                fullMapArray[drawInd,i,:,:] = fluxes

        ## note that the maps have the origin at the top
        ## so we have to flip the latitude array
        lats = np.flip(lats,axis=0)

    return fullMapArray, bestMapArray, lats, lons, waves



def plot_retrieved_map(fullMapArray,bestMapArray,lats,lons,waves,waveInd=3,degree=3,
                       saveName=None):
    percentiles = [5,50,95]
    mapLowMedHigh = np.percentile(fullMapArray,percentiles,axis=0)
    minflux=np.min(mapLowMedHigh[:,waveInd,:,:])
    maxflux=np.max(mapLowMedHigh[:,waveInd,:,:])
    londim = fullMapArray.shape[2]

    #replace the median map with the best fit map
    mapLowMedHigh[1]=bestMapArray

    fig, axArr = p.subplots(1,3,figsize=(22,5))
    for ind,onePercentile in enumerate(percentiles):
        map_day = mapLowMedHigh[ind][waveInd][:,londim//4:-londim//4]
        extent = np.array([np.min(lons)/np.pi/2.*180,np.max(lons)/np.pi/2.*180,np.min(lats)/np.pi*180,np.max(lats)/np.pi*180])
        plotData = axArr[ind].imshow(map_day, extent=extent,vmin=minflux,vmax=maxflux)
        cbar = fig.colorbar(plotData,ax=axArr[ind])
        cbar.set_label('Brightness')
        axArr[ind].set_ylabel('Latitude')
        axArr[ind].set_xlabel('Longitude')
        axArr[ind].set_title("{} %".format(onePercentile))
        #axArr[ind].show()

    fig.suptitle('Retrieved group map, n={}, {:.2f}$\mu$m'.format(degree,waves[waveInd]))
    p.savefig('plots/retrieved_maps/retrieved_map_{}_deg_{}_waveInd_{}.pdf'.format(saveName,degree,waveInd))


def get_map_and_plot(waveInd=3,degree=3,dataDir="data/sph_harmonic_coefficients_full_samples/hotspot/",
                     saveName=None,isspider=True):
    '''
    Plots spherical harmonic maps at one wavelength for 5th, 50th, and 95th percentile posterior samples

    Inputs
    ----------
    waveInd: int
        Index of the wavelength for which a map will be created
    degree: int
        Spherical harmonic degree to draw samples from
    dataDir: str
        Path to the directory containing the spherical harmonic coefficients

    Outputs
    -----------
    waves: array
        Wavelengths for the eigenspectra
    '''
    fullMapArray, bestMapArray, lats, lons, waves = retrieve_map_full_samples(degree=degree,dataDir=dataDir,isspider=isspider)
    plot_retrieved_map(fullMapArray,bestMapArray,lats,lons,waves,degree=degree,waveInd=waveInd,
                       saveName=saveName)
    return waves, lats, lons

def all_sph_degrees(waveInd=5):
    for oneDegree in np.arange(2,6):
        get_map_and_plot(waveInd=waveInd,degree=oneDegree)


def find_groups(dataDir,ngroups=4,degree=2,
                londim=100, latdim=100,
                trySamples=45,extent=0.5,sortMethod='avg',isspider=True):
    """
    Find the eigenspectra using k means clustering

    Parameters
    ----------
    ngroups: int
        Number of eigenspectra to group results into
    degree: int
        Spherical harmonic degree to draw samples from
    testNum: int
        Test number (ie. lightcurve number 1,2, etc.)
    trySamples: int
        Number of samples to find groups with
        All samples take a long time so this takes a random
        subset of samples from which to draw posteriors
    sortMethod: str
        Method to sort the groups returned by K means clustering
        None, will not sort the output
        'avg' will sort be the average of the spectrum
        'middle' will sort by the flux in the middle of the spectrum
    extent: time covered by the eclipse/phase curve.
        Sets what portion of the map is used for clustering (e.g. full planet or dayside only)
    """
    #samplesDir = "data/sph_harmonic_coefficients_full_samples"
    #dataDir = "{}/eclipse_lightcurve_test{}/".format(samplesDir,testNum)
    tmp = np.load("{}spherearray_deg_{}.npz".format(dataDir,degree))
    outDictionary = tmp['arr_0'].tolist()
    samples = outDictionary['spherical coefficients'] # output from eigencurves

    if trySamples>len(samples):
    	assert(trySamples<=len(samples)),("trySamples must be less than the total number of MCMC samples, "+str(len(samples)))

    eigenspectra_draws = []
    kgroup_draws = []
    uber_eigenlist=[[[[] for i in range(10)] for i in range(ngroups)] for i in range(trySamples)]

    if isspider:
        params0=sp.ModelParams(brightness_model='spherical') #megan added stuff
        params0.nlayers=20
        params0.t0=-2.21857/2.
        params0.per=2.21857567
        params0.a_abs=0.0313
        params0.inc=85.71
        params0.ecc=0.0
        params0.w=90.
        params0.rp=0.155313
        params0.a=8.863
        params0.p_u1=0.
        params0.p_u2=0.
        params0.degree=degree
        params0.la0=0.
        params0.lo0=0.
        waves=np.array([2.41,2.59,2.77,2.95,3.13,3.31,3.49,3.67,3.85,4.03])
    minlon=np.around(extent/2.*londim)
    #print(minlon)

    randomIndices = np.random.randint(0,len(samples),trySamples)
    for drawInd,draw in enumerate(samples[randomIndices]):
        ## Re-formatting here into a legacy system
        ## 1st dimension is wavelength
        ## 2nd dimensions is data (0th element = wavelength)
        ##                        (1: elements are spherical harmonic coefficients)
        if not isspider:
            inputArr = np.zeros([10,samples.shape[1]+1])
            inputArr[:,0] = np.array([2.41,2.59,2.77,2.95,3.13,3.31,3.49,3.67,3.85,4.03])
            inputArr[:,1:] = draw.transpose()

            waves, lats, lons, maps = eigenmaps.generate_maps(inputArr, N_lon=londim, N_lat=latdim)
            maps=maps[:,:,int(londim/2.-minlon):int(londim/2.+minlon)]

        #maps=np.zeros((np.shape(waves)[0],londim,latdim)) #full map

        else:
            maps=np.zeros((np.shape(waves)[0],latdim,int(minlon*2))) #only dayside
        #print(np.shape(maps))

            for i in np.arange(np.shape(waves)[0]):
                params0.sph=list(samples[drawInd,:,i])
                nla=latdim
                nlo=londim
                las = np.linspace(-np.pi/2,np.pi/2,nla)
                los = np.linspace(-np.pi,np.pi,nlo)
                fluxes = []
                for la in las:
                    row = []
                    for lo in los:
                        flux = sp.call_map_model(params0,la,lo)
                        row += [flux[0]]
                    fluxes += [row]
                fluxes = np.array(fluxes)
                lons, lats = np.meshgrid(los,las)
                #print(np.min(lats),np.min(lons),np.min(las),np.min(los))
                #lats, lons, maps = testgenmaps.spmap(inputArr,londim, latdim)
                #pdb.set_trace()
                maps[i,:,:] = fluxes[:,int(londim/2.-minlon):int(londim/2.+minlon)]

        kgroups = kmeans.kmeans(maps, ngroups)

        eigenspectra,eigenlist = bin_eigenspectra.bin_eigenspectra(maps, kgroups)

        eigenspectra_draws.append(eigenspectra)
        kgroup_draws.append(kgroups)
        for groupind in range(ngroups):
            for waveind in range(10):
                uber_eigenlist[drawInd][groupind][waveind]=eigenlist[groupind][waveind,:]

    if sortMethod is not None:
        eigenspectra_draws_final, kgroup_draws_final,uber_eigenlist_final = kmeans.sort_draws(eigenspectra_draws,
                                                                         kgroup_draws,uber_eigenlist,
                                                                         method=sortMethod)
    else:
        eigenspectra_draws_final, kgroup_draws_final,uber_eigenlist_final = eigenspectra_draws, kgroup_draws,uber_eigenlist
    return eigenspectra_draws_final, kgroup_draws_final,uber_eigenlist_final, maps

def show_spectra_of_groups(eigenspectra_draws,kgroup_draws,uber_eigenlist,waves,
                           saveName='kmeans',degree=None):
    """
    Calculate the mean and standard deviation of the spectra
    as well as the kgroups map
    Plot the mean and standard deviations of the spectra
    """
    #eigenspectra = np.mean(eigenspectra_draws, axis=0)
    #eigenerrs = np.std(eigenspectra_draws, axis=0)
    kgroups = np.mean(kgroup_draws, axis=0)

    allsamples=[[[] for i in range(np.shape(waves)[0])] for i in range(np.shape(uber_eigenlist)[1])]
    for x in range(np.shape(uber_eigenlist)[0]):
        for y in range(np.shape(uber_eigenlist)[1]):
            for z in range(np.shape(uber_eigenlist)[2]):
                allsamples[y][z]=np.concatenate((allsamples[y][z],uber_eigenlist[x][y][z]))

    eigenspectra=np.zeros((np.shape(allsamples)[0],np.shape(allsamples)[1]))
    eigenerrs=np.zeros((np.shape(allsamples)[0],np.shape(allsamples)[1]))
    for x in range(np.shape(allsamples)[0]):
        for y in range(np.shape(allsamples)[1]):
            eigenspectra[x,y]=np.mean(allsamples[x][y])
            eigenerrs[x,y]=np.std(allsamples[x][y])

    #print(np.shape(kgroups))
    #waves=np.array([2.41,2.59,2.77,2.95,3.13,3.31,3.49,3.67,3.85,4.03])
    #print(kgroups)
    #print(np.min(kgroups),np.max(kgroups))
    #print(np.around(np.min(kgroups)),np.around(np.max(kgroups)))
    counter=0
    colors=['b','g','orange','m']
    fig, ax = p.subplots()
    for spec, err in zip(eigenspectra, eigenerrs):
        ax.errorbar(waves, spec, err,label=('Group '+np.str(counter)),linewidth=2,marker='.',markersize=10,color=colors[counter])
        counter+=1
    ax.set_xlabel('Wavelength (micron)',fontsize=20)
    ax.set_ylabel('Fp/Fs',fontsize=20)
    ax.tick_params(labelsize=20,axis="both",right=True,top=True,width=1.5,length=5)
    ax.set_title('Eigenspectra')
    ax.legend(fontsize=15)

    Ngroup = eigenspectra_draws.shape[1]
    fig.savefig('plots/eigenmap_and_spec/{}_spectra_deg{}_grp_{}.pdf'.format(saveName,degree,Ngroup))

    return kgroups



def do_hue_maps(extent,maps,lons,lats,kgroups,ngroups,hueType='group'):
    #full_extent = np.array([np.min(lons),np.max(lons),np.min(lats),np.max(lats)])/np.pi*180 #for full map
    #full_extent = np.array([-90.,90.,-90.,90.]) #for dayside only
    full_extent = np.array([-extent/2.*360.,extent/2.*360.,-90.,90.])
    # londim, latdim = np.shape(maps)[1:]

    maps_mean = np.average(maps, axis=0)
    maps_error = np.std(maps, axis=0)

    cmap = cc.cm['isolum']
    cmap_grey = cc.cm['linear_grey_10_95_c0']
    cmap_grey_r = cc.cm['linear_grey_10_95_c0_r']
    # norm = Normalize(vmin=np.min(maps_mean), vmax=np.max(maps_mean))
    londim=100

    kround=np.around(kgroups)
    minlon=np.around(extent/2.*londim)

    contlons=lons[:,int(londim/2.-minlon):int(londim/2.+minlon)]
    contlats=lats[:,int(londim/2.-minlon):int(londim/2.+minlon)]

    if hueType == 'group':
        p.figure(figsize=(10,6.5))
        p.title('Eigengroups', fontsize=22)

        group_map = generate_map2d(hue_quantity=kround,
                                   lightness_quantity=maps_mean,
                                   hue_cmap=cmap,
                                   scale_min=10,
                                   scale_max=90)
        p.imshow(group_map, extent=full_extent, interpolation='gaussian')
        CS = p.contour(contlons/np.pi*180, contlats/np.pi*180, kround,
                       levels=np.arange(ngroups), colors='k', linestyles=['solid', 'dashed', 'dotted'])

        p.clabel(CS, inline=1, fmt='%1.0f', fontsize=12)

        p.xlabel(r'Longitude ($^\circ$)', fontsize=16)
        p.ylabel(r'Latitude ($^\circ$)', fontsize=16)
        p.setp(p.axes().get_xticklabels(), fontsize=16)
        p.setp(p.axes().get_yticklabels(), fontsize=16)

        cmap_group = cmap
        cNorm_group  = Normalize(vmin=0, vmax=ngroups-1)
        scalarMap_group = cm.ScalarMappable(norm=cNorm_group, cmap=cmap_group)

        cmap_flux = cmap_grey
        cNorm_flux  = Normalize(vmin=0, vmax=np.nanmax(maps_mean))
        scalarMap_flux = cm.ScalarMappable(norm=cNorm_flux, cmap=cmap_flux)

        bounds = np.linspace(-0.5, ngroups-0.5, ngroups+1)
        norm_group = BoundaryNorm(bounds, cmap_group.N)

        divider = make_axes_locatable(p.axes())
        ax2 = divider.append_axes("bottom", size="7.5%", pad=1)
        cb = colorbar.ColorbarBase(ax2, cmap=cmap_group, norm=norm_group, spacing="proportional", orientation='horizontal', ticks=np.arange(0, ngroups, 1), boundaries=bounds)
        cb.ax.xaxis.set_minor_formatter(FormatStrFormatter('%.1g'))
        cb.ax.tick_params(axis='x', labelsize=13)
        cb.ax.tick_params(axis='x', direction='inout',  top='off', bottom='off',
                          labeltop='on', labelbottom='off', labelsize=13, pad=-15)
        cb.ax.set_title('Group', y=1.35, fontsize=19)

        ax3 = divider.append_axes("bottom", size="7.5%", pad=0.75)
        cb = colorbar.ColorbarBase(ax3, cmap=cmap_flux, norm=cNorm_flux, orientation='horizontal')
        cb.ax.tick_params(axis='x', labelsize=13)
        cb.ax.set_title('Flux', y=1.35, fontsize=19)

        #for filetype in ['png', 'pdf']:
        #    p.savefig('HUEgroup_LUMflux_quadrant_deg6_group4.{}'.format(filetype), dpi=300, bbox_inches='tight')
    elif hueType == 'flux':
        p.figure(figsize=(10,6.5))
        p.title('Flux', fontsize=22)

        group_map = generate_map2d(hue_quantity=(maps_mean-np.min(maps_mean))/np.ptp(maps_mean),
                                   lightness_quantity=1-((maps_error*100.-np.min(maps_error*100.))/np.ptp(maps_error*100.)),
                                   hue_cmap=cmap,
                                   scale_min=10,
                                   scale_max=90)
        p.imshow(group_map, extent=full_extent, interpolation='gaussian')
        CS = p.contour(contlons/np.pi*180, contlats/np.pi*180, kround,
                       levels=np.arange(ngroups), colors='k', linestyles=['solid', 'dashed', 'dotted'])

        p.clabel(CS, inline=1, fmt='%1.0f', fontsize=12)

        p.xlabel(r'Longitude ($^\circ$)', fontsize=16)
        p.ylabel(r'Latitude ($^\circ$)', fontsize=16)
        p.setp(p.axes().get_xticklabels(), fontsize=16)
        p.setp(p.axes().get_yticklabels(), fontsize=16)

        cmap_flux = cmap
        cNorm_flux = Normalize(vmin=0, vmax=np.nanmax(maps_mean))
        scalarMap_flux = cm.ScalarMappable(norm=cNorm_flux, cmap=cmap_flux)

        cmap_stdev = cmap_grey_r
        cNorm_stdev  = Normalize(vmin=0, vmax=np.nanmax(maps_error*100.))
        scalarMap_stdev = cm.ScalarMappable(norm=cNorm_stdev, cmap=cmap_stdev)

        divider = make_axes_locatable(p.axes())
        ax2 = divider.append_axes("bottom", size="7.5%", pad=1)
        cb = colorbar.ColorbarBase(ax2, cmap=cmap_flux, norm=cNorm_flux, orientation='horizontal')
        cb.ax.tick_params(axis='x', labelsize=13)
        cb.ax.set_title('Flux', y=1.35, fontsize=19)

        ax3 = divider.append_axes("bottom", size="7.5%", pad=0.75)
        cb = colorbar.ColorbarBase(ax3, cmap=cmap_stdev, norm=cNorm_stdev, orientation='horizontal')
        cb.ax.tick_params(axis='x', labelsize=13)
        cb.ax.set_title('Uncertainty [\%]', y=1.35, fontsize=19)

        #for filetype in ['png', 'pdf']:
        #    p.savefig('HUEflux_LUMstdev_quadrant_deg6_group4.{}'.format(filetype), dpi=300, bbox_inches='tight')

    else:
        raise Exception("Unrecognized hueType {}".format(hueType))
