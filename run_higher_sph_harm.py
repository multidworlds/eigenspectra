
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
#import mapPCA
import bin_eigenspectra

from importlib import import_module
planet_name = 'HD189733b'
system = import_module('data.planet.{}'.format(planet_name))


# ### Import spectra and generate map

# Load lightcurve
stuff = np.load("data/input_lightcurves/eclipse_lightcurve_test1.npz")

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


inputLC3D = add_noise.get_lc()
noiseDict = add_noise.add_noise(inputLC3D)


# ### Fit eigencurves to lightcurve

# In[28]:


import importlib
importlib.reload(eigencurves)


# In[ ]:
norder=15

print("Fitting eigencurves now for order {}".format(norder))

spherearray = eigencurves.eigencurves(noiseDict,plot=True)
# spherearray is an array of wavelength x SH coefficents


# In[60]:


np.savez('spherearray_{}.npz'.format(norder),spherearray)

