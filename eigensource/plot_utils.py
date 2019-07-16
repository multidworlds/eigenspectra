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

def retrieve_map_full_samples(degree=3,dataDir="data/sph_harmonic_coefficients_full_samples/hotspot/"):
    tmp = np.load("{}spherearray_deg_{}.npz".format(dataDir,degree))
    outDictionary = tmp['arr_0'].tolist()
    
    londim = 100
    latdim = 100
    samples = outDictionary['spherical coefficients'] # output from eigencurves
    waves = outDictionary['wavelength (um)']
    
    randomIndices = np.random.randint(0,len(samples),39)
    nRandom = len(randomIndices)
    
    fullMapArray = np.zeros([nRandom,len(waves),londim,latdim])
    #SPIDERMAN stuff added by Megan
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
    
    for drawInd, draw in enumerate(samples[randomIndices]):
#         inputArr = np.zeros([len(waves),samples.shape[1]+1])
#         inputArr[:,0] = waves
#         inputArr[:,1:] = draw.transpose()
        
#         wavelengths, lats, lons, maps = eigenmaps.generate_maps(inputArr,
#                                                                 N_lon=londim, N_lat=latdim)
#         fullMapArray[drawInd,:,:,:] = maps
            #MEGAN ADDED STUFF
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
        
    
    return fullMapArray, lats, lons, waves
    

    
def plot_retrieved_map(fullMapArray,lats,lons,waves,waveInd=3,degree=3):
    percentiles = [5,50,95]
    mapLowMedHigh = np.percentile(fullMapArray,percentiles,axis=0)
    minflux=np.min(mapLowMedHigh[:,waveInd,:,:])
    maxflux=np.max(mapLowMedHigh[:,waveInd,:,:])
    londim = fullMapArray.shape[2]
    
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
    p.savefig('retrievedmap.png')

def get_map_and_plot(waveInd=3,degree=3,dataDir="data/sph_harmonic_coefficients_full_samples/hotspot/"):
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
    fullMapArray, lats, lons, waves = retrieve_map_full_samples(degree=degree,dataDir=dataDir)
    plot_retrieved_map(fullMapArray,lats,lons,waves,degree=degree,waveInd=waveInd)
    return waves, lats, lons

def all_sph_degrees(waveInd=5):
    for oneDegree in np.arange(2,6):
        get_map_and_plot(waveInd=waveInd,degree=oneDegree)
        

def find_groups(dataDir,ngroups=4,degree=2,
                londim=100, latdim=100,
                trySamples=45,extent=0.5,sortMethod='avg'):
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
        'middle' will sort by the flux in the middl of the spectrum
    extent: time covered by the eclipse/phase curve. 
        Sets what portion of the map is used for clustering (e.g. full planet or dayside only)
    """
    #samplesDir = "data/sph_harmonic_coefficients_full_samples"
    #dataDir = "{}/eclipse_lightcurve_test{}/".format(samplesDir,testNum)
    tmp = np.load("{}spherearray_deg_{}.npz".format(dataDir,degree))
    outDictionary = tmp['arr_0'].tolist()
    samples = outDictionary['spherical coefficients'] # output from eigencurves

    eigenspectra_draws = []
    kgroup_draws = []
    
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
        #inputArr = np.zeros([10,samples.shape[1]+1])
        #inputArr[:,0] = np.array([2.41,2.59,2.77,2.95,3.13,3.31,3.49,3.67,3.85,4.03])
        #inputArr[:,1:] = draw.transpose()

        #wavelengths, lats, lons, maps = eigenmaps.generate_maps(inputArr, N_lon=londim, N_lat=latdim)
        
        #maps=np.zeros((np.shape(waves)[0],londim,latdim)) #full map
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
            maps[i,:,:] = fluxes[:,int(londim/2.-minlon):int(londim/2.+minlon)]
            
        kgroups = kmeans.kmeans(maps, ngroups)

        eigenspectra = bin_eigenspectra.bin_eigenspectra(maps, kgroups)

        eigenspectra_draws.append(eigenspectra)
        kgroup_draws.append(kgroups)
    if sortMethod is not None:
        eigenspectra_draws_final, kgroup_draws_final = kmeans.sort_draws(eigenspectra_draws,
                                                                         kgroup_draws,
                                                                         method=sortMethod)
    else:
        eigenspectra_draws_final, kgroup_draws_final = eigenspectra_draws, kgroup_draws
    return eigenspectra_draws_final, kgroup_draws_final, maps
    
def show_spectra_of_groups(eigenspectra_draws,kgroup_draws,waves):
    """
    Calculate the mean and standard deviation of the spectra
    as well as the kgroups map
    Plot the mean and standard deviations of the spectra
    """
    eigenspectra = np.mean(eigenspectra_draws, axis=0)
    eigenerrs = np.std(eigenspectra_draws, axis=0)
    kgroups = np.mean(kgroup_draws, axis=0)
    print(np.shape(kgroups))
    #waves=np.array([2.41,2.59,2.77,2.95,3.13,3.31,3.49,3.67,3.85,4.03])
    #print(kgroups)
    print(np.min(kgroups),np.max(kgroups))
    print(np.around(np.min(kgroups)),np.around(np.max(kgroups)))
    counter=0
    colors=['b','g','orange','m']
    for spec, err in zip(eigenspectra, eigenerrs):
        p.errorbar(waves, spec, err,label=('Group '+np.str(counter)),linewidth=2,marker='.',markersize=10,color=colors[counter])
        counter+=1
    p.xlabel('Wavelength (micron)',fontsize=20)
    p.ylabel('Fp/Fs',fontsize=20)
    p.tick_params(labelsize=20,axis="both",right=True,top=True,width=1.5,length=5)
    p.title('Eigenspectra')
    p.legend(fontsize=15)
    
    return kgroups
    #p.show()
    #p.savefig('plots/eigenmap_and_spec/'+'quadrant_spectra_deg6_2groups_error_bars.pdf',bbox_inches='tight')