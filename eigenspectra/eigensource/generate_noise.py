# Import the libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
import os

try:
    import pynrc
except ModuleNotFoundError:
    print("ERROR in 'generate_noise': pynrc is not installed! See https://pynrc.readthedocs.io/en/latest/installation.html")

from copy import deepcopy
import pysynphot as S

HERE = os.path.abspath(os.path.split(__file__)[0])

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
    tSpec2.write(os.path.join(HERE, '../../data/instrument/snr_spec_f322w2.fits'),overwrite=True)
    tSpec2.write(os.path.join(HERE, '../../data/instrument/snr_spec_f322w2.csv'),overwrite=True)

    fig, ax = plt.subplots()

    plt.plot(tSpec2['wave'],tSpec2['sigma_ppm'])
    plt.ylim(300,900)

    fig.savefig(os.path.join(HERE, '../../data/instrument/snr_spectrum.pdf'))
