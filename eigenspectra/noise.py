"""
Make Noise
----------
(Replaces the files `add_noise` and `generate_noise` from ``eigensource``)
"""

import numpy as np
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
import pdb
import os

try:
    import pynrc
except ModuleNotFoundError:
    print("ERROR in 'generate_noise': pynrc is not installed! See https://pynrc.readthedocs.io/en/latest/installation.html")

from copy import deepcopy
import pysynphot as S

HERE = os.path.abspath(os.path.split(__file__)[0])

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
dat = Table.read(os.path.join(HERE, '../data/instrument/snr_spec_f322w2.fits'))

def get_lc(practiceVersion=False,
          inputFile='data/input_lightcurves/eclipse_lightcurve_test2.npz'):
    """
    Get the input lightcurves from the NPZ files and put them in the format
    that ``add_noise`` is expecting

    Parameters
    ----------
    inputFile: str
          path to the output NPZ file
    practiceVersion: bool
          Use a practice version?
    """

    ### EXAMPLE ONLY
    if practiceVersion == True:
        lcDat = np.load(os.path.join(HERE, '../data/input_lightcurves/eclipse_lightcurve_test0.npz'))
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

    Parameters
    ----------
    fluxTWave: numpy.array
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

    Returns
    -------
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
                t.write(os.path.join(HERE, '../data/output_lightcurves/{}.csv'.format(baseName)),overwrite=True)

            outDict["flux (ppm)"][:,waveInd] = t['flux (ppm)']
            outDict["flux err (ppm)"][:,waveInd] = t['flux err (ppm)']


            if doPlots == True:
                fig, ax = plt.subplots()
                ax.errorbar(t['time (days)'],t['flux (ppm)'],fmt='o',
                            yerr=t['flux err (ppm)'])

                fig.savefig(os.path.join(HERE, '../data/output_lightcurves/plots/{}.pdf'.format(baseName)))


        else:
            print('No SNR information for wavelength {} um'.format(waveMid))
            outDict["flux (ppm)"][:,waveInd] = np.nan
            outDict["flux err (ppm)"][:,waveInd] = np.nan


    if doPlots == True:
        plt.close('all')

    return outDict


################################################################################


def sum_spectrum(img,ap=4,center=None):
    """Simply sums the spectrum along spatial direction"""
    if center is None:
        center = img.shape[0]/2
    subImg = img[(center-ap):(center+ap),:]

    return np.sum(subImg,axis=0)


def make_snr_spectrum():
    """
    Make a SNR spectrum and save an output data file and plot.
    """

    bp_k = S.ObsBandpass('johnson,k')
    sp = pynrc.stellar_spectrum('K0V',5.541,'vegamag',bp_k,)

    nrc = pynrc.NIRCam('F322W2', pupil='GRISM0', ngroup=3, nint=100, read_mode='RAPID',
                      wind_mode='STRIPE',xpix=2048,ypix=64)
    det = nrc.Detectors[0]
    wave, psfImg = nrc.gen_psf(sp=sp)
    oneDSpec = np.sum(psfImg,axis=0)
    pix_noise = det.pixel_noise(fsrc=psfImg,fzodi=nrc.bg_zodi(zfact=1.0))
    varImg = pix_noise**2
    SNR = np.sum(psfImg) / np.sqrt(np.sum(varImg))
    print(' Broadband noise (ppm) = {0}'.format(1e6/SNR))

    ## SNR spectrum
    SNRSpec = np.sum(psfImg,axis=0) / np.sqrt(np.sum(varImg,axis=0))
    tSpec = Table()
    tSpec['wave'] = wave
    tSpec['sigma_ppm'] = 1e6/SNRSpec

    keepWave = (tSpec['wave'] > 2.4) & (tSpec['wave'] <= 4.02)
    tSpec2 = tSpec[keepWave]

    tSpec2.meta = {'filter':'F322W2W','TEXP':det.time_total}
    tSpec2.write(os.path.join(HERE, '../data/instrument/snr_spec_f322w2.fits'),overwrite=True)
    tSpec2.write(os.path.join(HERE, '../data/instrument/snr_spec_f322w2.csv'),overwrite=True)

    fig, ax = plt.subplots()

    plt.plot(tSpec2['wave'],tSpec2['sigma_ppm'])
    plt.ylim(300,900)

    fig.savefig(os.path.join(HERE, '../data/instrument/snr_spectrum.pdf'))
