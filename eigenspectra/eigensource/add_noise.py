import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import pdb

## Constants in data structure
Ncolumns = 4
waveColumn = 0
dWaveColumn = 1
timeColumn = 2
fluxColumn = 3

## Put in 
## Placeholder matrix until we have one produced
exNWave, exNtime = 10, 3000
fluxTimeWavelengthExample = np.ones([Ncolumns,exNtime,exNWave])

for waveInd in np.arange(exNWave):
    fluxTimeWavelengthExample[timeColumn,:,waveInd] = np.linspace(-2000,2000,exNtime)
for tInd in np.arange(exNtime):
    fluxTimeWavelengthExample[waveColumn,tInd,:] = np.linspace(2.4,4,exNWave)


## Get the SNR file
dat = Table.read('data/instrument/snr_spec_f322w2.fits')

def get_lc(practiceVersion=False,
          inputFile='data/input_lightcurves/eclipse_lightcurve_test2.npz'):
    """ 
    Get the input lightcurves from the NPZ files and put them in the format
          that add_noise is expecting
          
    Inputs
    --------
    inputFile: str
          path to the output NPZ file
    practiceVersion: bool
          Use a practice version?
    """
    
    ### EXAMPLE ONLY
    if practiceVersion == True:
        lcDat = np.load('data/input_lightcurves/eclipse_lightcurve_test0.npz')
    else:
        lcDat = np.load(inputFile)
    
    nWave = len(lcDat['wl'])
    nTime = len(lcDat['time'])
    timeArray = lcDat['time'] * 3600. * 24.
    
    if practiceVersion == True:
        waves = np.linspace(2.4,4,nWave)
        dWaves = np.ones_like(waves) * (waves[1] - waves[0])
        timeArray = np.linspace(0.5,0.6,nTime) * 3600. * 24.
    else:
        waves = lcDat['wl']
        dWaves = lcDat['dwl']
    
    input3D = np.ones([Ncolumns,nTime,nWave])
    for waveInd,oneWave in enumerate(waves):
        input3D[waveColumn,:,waveInd] = oneWave
        input3D[dWaveColumn,:,waveInd] = dWaves[waveInd]
        input3D[timeColumn,:,waveInd] = timeArray
        
        input3D[fluxColumn,:,waveInd] = lcDat['lightcurve'][:,waveInd] * 1e6
    
    return input3D
    
def noise_on_input_lc(practiceVersion=False):
    input3D = get_lc(practiceVersion=False)
    add_noise(input3D)

def add_noise(fluxTWave,preserveInput=True,nEclipses=5,
              includeSystematic=False,renormalize=True,
              doPlots=False,writeCSVs=False):
    """ 
    Takes a series of light curves (one per wavelength) and
    Adds noise to them. First it bins to the integration time in the data
    Then it creates
    
    Inputs
    -------
    fluxTWave: numpy array
        A 3D array with z,y,x axes as n_columns x n_times x wavelength
        The 4 n_columns are
         - wavelength in microns (repeated)
         - wavelength width in microns (repeated)
         - time (seconds)
         - flux in Fp/F* (ppm)
        There are t rows 
    
    preserveInput: bool
        Preserve the input array? Default is true. This will not add any
        random noise to the data and just preserve the input
        The rationale is that this way your posteriors should have a median close to the
        true Input. Otherwise, you'd have to run the fits multiple times to test if the 
        median posterior was centered on your true input solution
    
    nEclipses: int
        How many eclipses are combined together?
    
    renormalize: bool
        Renormalize the time series so that the minimum flux is 1.0?
        This was designed to match the Spiderman normalization
    
    doPlots: bool
        Save plots of the light curves? Makes one light curve per wavelength
    
    writeCSVs: bool
        Save the light curves as CSVs?
    
    Output
    --------
    Dictionary of information
    
    """
    

    texp = dat.meta['TEXP'] ## seconds
    
    nColumns, nTime, nWave = fluxTWave.shape

    
    minTime, maxTime = np.min(fluxTWave[timeColumn,:,:]), np.max(fluxTWave[timeColumn,:,:])
    
    
    timeStarts = np.arange(minTime,maxTime - texp,texp)
    timeEnds = timeStarts + texp
    timeMid = (timeStarts + timeEnds)/2.
    
    outDict = {"time (days)":timeMid / (3600. * 24.),"wavelength (um)":fluxTWave[waveColumn,0,:],
               "flux (ppm)": np.zeros([len(timeMid),nWave]),
               "flux err (ppm)": np.zeros([len(timeMid),nWave])}
    
    for waveInd in np.arange(nWave):
        fluxArray, errFluxArr = [], []
        waveMid = fluxTWave[waveColumn,0,waveInd]
        dWave = fluxTWave[dWaveColumn,0,waveInd]
        wavePts = (dat['wave'] > waveMid - dWave) & (dat['wave'] <= waveMid + dWave)
        
        if np.sum(wavePts) >= 1:
            
            combinedErr = np.sqrt(np.sum(dat['sigma_ppm']**2)) / np.sum(wavePts)
            
            
            for tInd,oneTime in enumerate(timeMid):
                timeArray = fluxTWave[timeColumn,:,waveInd]
                inPts = (timeArray > timeStarts[tInd]) & (timeArray <= timeEnds[tInd])
                binFlux = np.mean(fluxTWave[fluxColumn,inPts,waveInd])
                
                fluxArray.append(binFlux)
                errFluxArr.append(combinedErr)
                
            
            t = Table()
            t['time (days)'] = timeMid / (3600. * 24.)
            
            if preserveInput == True:
                noiseVectors = 0
            else:
                raise NotImplementedError
            
            t['flux (ppm)'] = np.array(fluxArray) + noiseVectors
            t['flux err (ppm)'] = np.array(errFluxArr) / np.sqrt(nEclipses)
            
            baseName = 'tseries_{:05.0f}_nm'.format(waveMid * 1000.)
            
            if renormalize == True:
                t['flux (ppm)'] = 1e6 * t['flux (ppm)'] / np.min(t['flux (ppm)'])
            
            if writeCSVs:
                t.write('data/output_lightcurves/{}.csv'.format(baseName),overwrite=True)
            
            outDict["flux (ppm)"][:,waveInd] = t['flux (ppm)']
            outDict["flux err (ppm)"][:,waveInd] = t['flux err (ppm)']
            
            
            if doPlots == True:
                fig, ax = plt.subplots()
                ax.errorbar(t['time (days)'],t['flux (ppm)'],fmt='o',
                            yerr=t['flux err (ppm)'])
                
                fig.savefig('data/output_lightcurves/plots/{}.pdf'.format(baseName))
            
            
        else:
            print('No SNR information for wavelength {} um'.format(waveMid))
            outDict["flux (ppm)"][:,waveInd] = np.nan
            outDict["flux err (ppm)"][:,waveInd] = np.nan
    
        
    if doPlots == True:
        plt.close('all')
    
    return outDict
    