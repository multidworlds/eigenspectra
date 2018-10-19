"""
gen_lightcurves.py
------------------
Methods for generating synthetic multi-wavelength exoplanet
secondary eclipse lightcurves

Example
-------
>>> from gen_lightcurves import create_lightcurves_with_starry
>>> time, lamlo, dlamlo, lightcurve = create_lightcurves_with_starry()

"""


import numpy as np
from scipy.stats import binned_statistic
import healpy as hp
import matplotlib.pyplot as plt

def downbin_spec(specHR, lamHR, lamLR, dlam=None):
    """
    Re-bin spectum to lower resolution using :py:obj:`scipy.binned_statistic`
    with ``statistic = 'mean'``. This is a "top-hat" convolution.

    Note
    ----
    This function is from `coronagraph` (https://coronagraph.readthedocs.io/en/latest/index.html),
    developed by J. Lustig-Yaeger.

    Parameters
    ----------
    specHR : array-like
        Spectrum to be degraded
    lamHR : array-like
        High-res wavelength grid
    lamLR : array-like
        Low-res wavelength grid
    dlam : array-like, optional
        Low-res wavelength width grid

    Returns
    -------
    specLR : :py:obj:`numpy.ndarray`
        Low-res spectrum
    """

    if dlam is None:
        ValueError("Please supply dlam in downbin_spec()")

    # Reverse ordering if wl vector is decreasing with index
    if len(lamLR) > 1:
        if lamHR[0] > lamHR[1]:
            lamHI = np.array(lamHR[::-1])
            spec = np.array(specHR[::-1])
        if lamLR[0] > lamLR[1]:
            lamLO = np.array(lamLR[::-1])
            dlamLO = np.array(dlam[::-1])

    # Calculate bin edges
    LRedges = np.hstack([lamLR - 0.5*dlam, lamLR[-1]+0.5*dlam[-1]])

    # Call scipy.stats.binned_statistic()
    specLR = binned_statistic(lamHR, specHR, statistic="mean", bins=LRedges)[0]

    return specLR

def planck(temp, lam):
    """
    Calculates the Planck blackbody radiance

    Parameters
    ----------
    temp : float or array-like
        Temperature [K]
    lam : array-like or float
        Wavelength(s) [um]

    Returns
    -------
    Blam : float or array-like
        Blackbody radiance [W/m**2/um/sr]

    Note
    ----
    Multiply by :math:`\pi` to get spectral flux density
    """
    h = 6.62607e-34       # Planck constant (J * s)
    c = 2.998e8           # Speed of light (m / s)
    k = 1.3807e-23        # Boltzmann constant (J / K)
    wav = lam * 1e-6
    # Returns B_lambda [W/m^2/um/sr]
    return 1e-6 * (2. * h * c**2) / (wav**5) / (np.exp(h * c / (wav * k * temp)) - 1.0)

def construct_lam(lammin, lammax, Res=None, dlam=None):
    """
    Construct a wavelength grid by specifying either a resolving power (`Res`)
    or a bandwidth (`dlam`)

    Note
    ----
    This function is from `coronagraph` (https://coronagraph.readthedocs.io/en/latest/index.html),
    developed by J. Lustig-Yaeger.

    Parameters
    ----------
    lammin : float
        Minimum wavelength [microns]
    lammax : float
        Maximum wavelength [microns]
    Res : float, optional
        Resolving power (lambda / delta-lambda)
    dlam : float, optional
        Spectral element width for evenly spaced grid [microns]

    Returns
    -------
    lam : float or array-like
        Wavelength [um]
    dlam : float or array-like
        Spectral element width [um]
    """

    # Keyword catching logic
    goR = False
    goL = False
    if ((Res is None) and (dlam is None)) or (Res is not None) and (dlam is not None):
        print("Error in construct_lam: Must specify either Res or dlam, but not both")
    elif Res is not None:
        goR = True
    elif dlam is not None:
        goL = True
    else:
        print("Error in construct_lam: Should not enter this else statment! :)")
        return None, None

    # If Res is provided, generate equal resolving power wavelength grid
    if goR:

        # Set wavelength grid
        dlam0 = lammin/Res
        dlam1 = lammax/Res
        lam  = lammin #in [um]
        Nlam = 1
        while (lam < lammax + dlam1):
            lam  = lam + lam/Res
            Nlam = Nlam +1
        lam    = np.zeros(Nlam)
        lam[0] = lammin
        for j in range(1,Nlam):
            lam[j] = lam[j-1] + lam[j-1]/Res
        Nlam = len(lam)
        dlam = np.zeros(Nlam) #grid widths (um)

        # Set wavelength widths
        for j in range(1,Nlam-1):
            dlam[j] = 0.5*(lam[j+1]+lam[j]) - 0.5*(lam[j-1]+lam[j])

        #Set edges to be same as neighbor
        dlam[0] = dlam0#dlam[1]
        dlam[Nlam-1] = dlam1#dlam[Nlam-2]

        lam = lam[:-1]
        dlam = dlam[:-1]

    # If dlam is provided, generate evenly spaced grid
    if goL:
        lam = np.arange(lammin, lammax+dlam, dlam)
        dlam = dlam + np.zeros_like(lam)

    return lam, dlam

def create_lightcurves_with_starry(plot_input_hp_maps = False, plot_lightcurves = True,
                                   plot_points_on_map_spec = True, plot_diagnostic = True,
                                   save_output = False):
    """
    Creates toy multi-wavelength secondary eclipse lightcurves of HD189733 b

    Returns
    -------
    time : numpy.array
        1d array of lightcurve times in days
    lamlo : numpy.array
        1d array of wavelengths in microns
    dlamlo : numpy.array
        1d array of wavelength bin widths
    lightcurve : numpy.ndarray
        2d array of multi-wavelength lightcurves with
        shape ``(len(time), len(lamlo))``
    """

    # Only this function should import starry so it doesn't
    # break if you don't have it
    import starry

    # Load in 3D data from Kat
    """
    data = np.load("data/maps/mystery_map.npy")
    data = data[:,1:][:,::-1]
    wldata = np.load("data/maps/wavelength.npy")
    wldata = wldata[::-1]
    """
    file = np.load("data/maps/mystery_map1.npz")  # Same file but w/wl included
    data = file["spaxels"]
    wldata = file["wl"]

    # Construct low res wavelength grid
    lammin = 2.41
    lammax = 3.98
    dlami = 0.18   # This gives 10 wavelengths
    lamlo, dlamlo = construct_lam(lammin, lammax, dlam=dlami)
    Nlamlo = len(lamlo)

    # Set HealPy pixel numbers
    Nside = 8
    Npix = hp.nside2npix(Nside)

    # Define empty 2d array for spaxels
    spec2d = np.zeros((Npix, Nlamlo))

    # Loop over pixels filling with blackbody
    for i in range(hp.nside2npix(Nside)):
        spec2d[i,:] = downbin_spec(data[i, :], wldata, lamlo, dlam = dlamlo)

    # Make diagnostic plots
    if plot_input_hp_maps:
        indices = [0,5,-1]
        for i in indices:
            hp.mollview(spec2d[:,i], title=r"%0.2f $\mu$m" %lamlo[i])
            plt.show()

    # Set starry params
    lmax = 18
    planet = starry.kepler.Secondary(lmax=lmax, nwav=Nlamlo)

    # HD189 parameters
    planet.r = 0.155313
    L = 0.00344
    planet.L = L * np.ones(Nlamlo)
    planet.inc = 85.71
    planet.a = 8.863
    planet.prot = 2.21857567
    planet.porb = 2.21857567
    planet.tref = -2.21857567 / 2.
    planet.ecc = 0.0
    planet.w = 90

    # Calculate disk-integrated Fp/Fs -- This should be the secondary eclipse depth,
    # but it must be consistent with the individual map fluxes
    FpFslr = np.mean(spec2d, axis=0)

    # Loop over wavelengths loading map into starry
    for i in range(Nlamlo):
        planet.load_healpix(spec2d[:, i], lmax = lmax, nwav = i)

    # Get planet the spherical harmonic map vector. This is a vector of the
    # coefficients of the spherical harmonics
    y = np.array(planet.y)

    # Get the total flux received by the observer from the map.
    f = np.array(planet.flux())

    # Scale the planet luminosoty by factors derived from the original map
    # WARNING: THIS IS PART OF THE HACK B/C OF STARRY ISSUE
    planet[:, :] = y
    fac1 = planet(0, -0.5).transpose()[0] / spec2d[0]
    fac2 = planet(0, 0.5).transpose()[0] / spec2d[-1]
    fac = np.mean((fac1, fac2), axis=0)
    planet.L = 1 / fac
    if True:
        plt.plot(lamlo, spec2d[0])
        plt.plot(lamlo, spec2d[-1])
        plt.plot(lamlo, planet(0, -0.5).transpose()[0] * planet.L, 'o', color="C0", ms=2)
        plt.plot(lamlo, planet(0, 0.5).transpose()[0] * planet.L, 'o', color="C1", ms=2)
        plt.xlabel("Wavelength [um]")
        plt.ylabel("Fp/Fs")
        plt.show()

    # Get the map intensity across the planet
    nx, ny = 300, 300
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    I = np.zeros((nx, ny, Nlamlo))
    for i in range(nx):
        I[i] = planet(x=X[i], y=Y[i])

    # Plot the starry map during secondary eclipse
    if False:
        iwavelength = -1
        fig, ax = plt.subplots(figsize = (4,4))
        ax.imshow(I[:,:,iwavelength],origin="lower")
        ax.axis('off')
        plt.show()

    # Create a generic star with starry
    star = starry.kepler.Primary(nwav=Nlamlo)

    # Set wavelength-dependent limb darkening params, which don't matter in
    # Secondary eclipse
    star[1] = 0.40 * np.ones(Nlamlo)
    star[2] = 0.26 * np.ones(Nlamlo)

    # Create starry system with star and planet
    system = starry.kepler.System(star, planet)

    # Create time grid in days (secondary eclipse is at 0.0)
    time = np.linspace(-0.1, 0.1, 10000)

    # Calculate lightcurve in starry
    system.compute(time)
    lightcurve = system.lightcurve / system.lightcurve[0]

    # THIS IS A HACK TO GET REASONABLE DEPTHS WHILE STARRY IS MESSING UP
    # NEEDS TO BE FIXED
    lightcurve = (lightcurve / 200.0)
    lightcurve = lightcurve + (1 - np.max(lightcurve))

    if plot_lightcurves:
        # Make Plot
        fig, ax = plt.subplots(1, figsize=(14, 5))
        ax.set_xlabel('Time [days]')
        ax.set_ylabel('Relative Flux')
        for i in range(len(lamlo)):
            # Subtract off minimum (which is the bottom of the eclipse) to look
            # like traditional form
            lc = lightcurve[:,i] - np.min(lightcurve[:,i])
            ax.plot(time, lc, c = "C%i" %(i%9), label = r"%.2f $\mu$m" %(lamlo[i]))
        ax.legend(ncol = 2)

    if plot_points_on_map_spec:

        # Which points to plot
        x1, y1 = 100, 200
        x2, y2 = 100, 100
        x3, y3 = 200, 200
        x4, y4 = 200, 100

        # Which wavelength to plot map at
        iwavelength = 0

        fig, ax = plt.subplots(1, 2, figsize = (10,4))
        ax[0].imshow(I[:,:,iwavelength],origin="lower")
        ax[0].axis('off')

        # Plot points
        ax[0].plot(x1,y1, "o", c="C0", ms = 10)
        ax[0].plot(x2,y2, "o", c="C1", ms = 10)
        ax[0].plot(x3,y3, "o", c="C2", ms = 10)
        ax[0].plot(x4,y4, "o", c="C3", ms = 10)

        # Plot FpFs
        ax[1].set_xlabel(r"Wavelength [$\mu$m]")
        ax[1].set_ylabel(r"Relative Flux")
        ax[1].plot(lamlo, I[x1,y1,:], c="C0")
        ax[1].plot(lamlo, I[x2,y2,:], c="C1")
        ax[1].plot(lamlo, I[x3,y3,:], c="C2")
        ax[1].plot(lamlo, I[x4,y4,:], c="C3")

        plt.show()

    if plot_diagnostic:

        # Create figure
        fig, ax = plt.subplots(figsize = (12, 4))
        ax.set_ylabel("Fp/Fs")
        ax.set_xlabel("Wavelength [$\mu$m]")

        # Loop over input spaxels plotting each spectrum (creates thick lines as
        # they overplot)
        for i in range(Npix):
            if i == 0:
                ax.plot(lamlo, spec2d[i,:], color = "k", label = "Inputs")
            else:
                ax.plot(lamlo, spec2d[i,:], color = "k")

        draws = 100
        # Loop over number of draws
        for k in range(draws):
            # Draw a random intensity from the planet
            i = np.random.randint(0,high=nx)
            j = np.random.randint(0,high=nx)
            # Plot it
            if k == 0:
                ax.plot(lamlo, I[i,j,:], color = "C1", lw = 0.5,
                        label = "starry samples")
            else:
                ax.plot(lamlo, I[i,j,:], color = "C1", lw = 0.5)

        ax.legend()

        plt.show()

    if save_output:
        # Save generated lightcurve
        np.savez("data/input_lightcurves/eclipse_lightcurve_test1.npz", time = time, wl = lamlo,
                                    dwl = dlamlo, lightcurve = lightcurve)


    return time, lamlo, dlamlo, lightcurve
