"""

"""
# coding: utf-8

# In[7]:


# Import functions
import numpy as np
import matplotlib.pyplot as p
#get_ipython().run_line_magic('matplotlib', 'inline')

import eigensource.add_noise as add_noise
import eigencurves
import eigenmaps
import kmeans
from sys import argv
import os
#import mapPCA
import bin_eigenspectra

from importlib import import_module

## Set the planet parameters

planet_name = 'HD189733b'
system = import_module('data.planet.{}'.format(planet_name))


def run_lc_noise_and_fit(norder=3,
                         usePath="data/input_lightcurves/eclipse_lightcurve_test1.npz",afew=10):
    """
    Add error bars to a forward model and then run eigencurves fitting to retrieve the map
    
    Parameters:
    -----------
    norder: int
        Number of the spherical harmonic to fit
    usePath: str
        Path to the npz file for light curves
    afew: int
        How many eigencurves to fit for (if >=10, the best fitting algorithm will run and select the number of eigencurves for you)
    
    Outputs:
    -----------
    None: all data are saved to data/sph_harmonic_coefficients_full_samples
    """
    # In[60]:
    ## Grab the name of the lightcurve
    lcName = os.path.splitext(os.path.basename(usePath))[0]
    
    saveDir = "data/sph_harmonic_coefficients_full_samples/" + lcName
    if os.path.exists(saveDir) == False:
        os.mkdir(saveDir)
    outputNPZ = '{}/spherearray_deg_{}.npz'.format(saveDir,norder)
    
    if os.path.exists(outputNPZ):
        print("Found the previously-run file {}. Now exiting".format(outputNPZ))
        return
    else:
        print("No previous run found, so running MCMC.")
        print("This can take a long time, especially for higher spherical harmonic orders")
        
        # ### Import spectra and generate map
        # Load lightcurve
        stuff = np.load(usePath)

        # Parse File
        lightcurve = stuff["lightcurve"]
        wl = stuff["wl"]
        dwl = stuff["dwl"]
        time = stuff["time"]

        # Make Plot
        fig, ax = p.subplots(1, figsize=(14, 5))
        ax.set_xlabel('Time [days]')
        ax.set_ylabel('Relative Flux')
        for i in range(len(wl)):
            lc = lightcurve[:,i] - np.min(lightcurve[:,i])
            ax.plot(time, lc+1, c = "C%i" %(i%9), label = r"%.2f $\mu$m" %(wl[i]))
        ax.legend(fontsize = 16, ncol = 2)
        #p.show()

        # ### Add Noise

        # In[12]:
        inputLC3D = add_noise.get_lc(inputFile=usePath)
        noiseDict = add_noise.add_noise(inputLC3D)


        # ### Fit eigencurves to lightcurve
        print("Fitting eigencurves now for order {}".format(norder))
        spherearray = eigencurves.eigencurves(noiseDict,plot=False,degree=norder,afew=afew)
        # spherearray is an array of wavelength x SH coefficents
    
        np.savez(outputNPZ,spherearray)

if __name__ == "__main__":
    """ If running on the command line, set the norder and usePath parameters
    """
    if len(argv) < 3:
        usePath = "data/input_lightcurves/eclipse_lightcurve_test1.npz"
        print("No lightcurve specified, using {}".format(usePath))
    else:
        usePath = argv[2]

    if len(argv) < 2:
        norder=3
        print("No order specified, using {}".format(norder))
    else:
        norder = int(argv[1])

    run_lc_noise_and_fit(norder=norder,usePath=usePath)
