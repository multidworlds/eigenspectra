"""
gen_lightcurves.py
------------------
Methods for generating synthetic multi-wavelength exoplanet
secondary eclipse lightcurves

Example
-------
>>> from gen_lightcurves import *
>>> lamhr, spaxels = prep_spectral_hotspot_map()
>>> time, lam, dlam, lcurves = create_lightcurves_with_starry(lamhr, spaxels)

"""


import numpy as np
from scipy.stats import binned_statistic
from scipy import signal
import healpy as hp
import matplotlib.pyplot as plt
import os, sys
import starry

HERE = os.path.dirname(os.path.abspath(__file__))

def expand_hp(healpix_map, lmax):
    """
    Expand a Healpix ring-ordered map in spherical harmonics up to degree `lmax`.

    """
    # Get the complex spherical harmonic coefficients
    alm = hp.sphtfunc.map2alm(healpix_map, lmax=lmax)

    # Convert them to real coefficients
    ylm = np.zeros(lmax ** 2 + 2 * lmax + 1, dtype='float')
    i = 0
    for l in range(0, lmax + 1):
        for m in range(-l, l + 1):
            j = hp.sphtfunc.Alm.getidx(lmax, l, np.abs(m))
            if m < 0:
                ylm[i] = np.sqrt(2) * (-1) ** m * alm[j].imag
            elif m == 0:
                ylm[i] = alm[j].real
            else:
                ylm[i] = np.sqrt(2) * (-1) ** m * alm[j].real
            i += 1

    # Instantiate a starry map so we can rotate it to the
    # correct orientation
    map = starry.Map(lmax=lmax)
    map[:, :] = ylm
    map.axis = [1, 0, 0]
    map.rotate(90.0);
    map.axis = [0, 0, 1]
    map.rotate(180.0);
    map.axis = [0, 1, 0]
    map.rotate(90.0);
    norm = 2 / np.sqrt(np.pi)
    return np.array(map.y / norm)


def downbin_spec(specHR, lamHR, lamLR, dlam=None):
    """
    Re-bin spectum to lower resolution using :py:obj:`scipy.binned_statistic`
    with ``statistic = 'mean'``. This is a "top-hat" convolution.

    Note
    ----
    This function is from `coronagraph` (https://jlustigy.github.io/coronagraph/),
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

def prep_map1():
    """
    Prepare toy map #1
    """

    # Load in 3D data from Kat
    path = os.path.join(HERE, "data/maps/mystery_map1.npz")
    file = np.load(path)  # Same file but w/wl included
    spaxels = file["spaxels"]
    lam = file["wl"]

    return lam, spaxels

def prep_map2():
    """
    Prepare toy map #2
    """

    # Load in 3D data from Kat
    path = os.path.join(HERE, "data/maps/mystery_map2.npz")
    file = np.load(path)  # Same file but w/wl included
    spaxels = file["spaxels"]
    lam = file["wl"]

    return lam, spaxels

def prep_blackbody_hotspot_map(phi0 = np.pi, theta0 = 0.0, ds = 25.0,
                               Tp1 = 900., Tp2 = 1200., Ts = 2550., RpRs = 0.155313,
                               lammin = 1.0, lammax = 5.0,
                               Nlam = 1000, Nside = 16):
    """
    Generate a toy spaxel map with a hotspot

    Parameters
    ----------
    phi0 : float
        Latitude of hot spot center [radians]
    theta0 : float
        Longitude of hot spot center [radians]
    ds : float
        Angular radius of hotspot [degrees]
    Tp1 : float
        Temperature outside hotspot [K]
    Tp2 : float
        Temperature inside hotspot [K]
    Ts : float
        Stellar effective temperature [K]
    RpRs : float
        Planet radius relative to stellar radius

    Returns
    -------
    lamhr : numpy.array
        Wavelength grid [microns]
    spaxels : numpy.ndarray
        Pixel spectra (2d)

    Example
    -------
    >>> lam, spaxels = prep_blackbody_hotspot_map()
    """

    # Create hi-res wavelength grid
    lamhr = np.linspace(lammin - 0.5, lammax + 0.5, Nlam)

    # Create blackbody spectra at two temperatures
    Bp1 = planck(Tp1, lamhr)     # Blackbody 1
    Bp2 = planck(Tp2, lamhr)     # Blackbody 2
    Bs = planck(Ts, lamhr)       # Stellar blackbody

    # Calculate the planet-star flux ratio for both "surfaces"
    FpFs1 = (RpRs**2) * (Bp1 / Bs)
    FpFs2 = (RpRs**2) * (Bp2 / Bs)

    # Get number of pixels/spaxels
    Npix = hp.nside2npix(Nside)

    # Calc the latitude and longitude of each hpix
    thetas, phis = hp.pix2ang(Nside, np.arange(Npix))
    thetas = thetas - np.pi / 2
    phis = phis - np.pi

    # Define empty 2d array for spaxels
    spaxels = np.zeros((Npix, Nlam))

    # Loop over pixels filling with blackbody
    for i in range(hp.nside2npix(Nside)):

        # Relabel variables for below equation
        phi1, theta1 =  thetas[i], phis[i]
        dtheta = theta0 - theta1

        # Calculate the angular distance between two points on sphere
        ang = np.arccos(np.sin(phi1) * np.sin(phi0) + np.cos(phi1) * np.cos(phi0) * np.cos(dtheta)) * 180 / np.pi

        # Use a different spectrum inside and outside of the great circle defined by dang
        if ang > ds:
            # Outside hotspot
            spaxels[i,:] = FpFs1
        else:
            # Within hotspot
            spaxels[i,:] = FpFs2

    return lamhr, spaxels

def prep_spectral_hotspot_map(phi0 = np.pi, theta0 = 0.0, ds = 25.0,
                              Nside = 16):
    """
    Generate a toy spaxel map with a hotspot

    Parameters
    ----------
    phi0 : float
        Latitude of hot spot center [radians]
    theta0 : float
        Longitude of hot spot center [radians]
    ds : float
        Angular radius of hotspot [degrees]

    Returns
    -------
    lamhr : numpy.array
        Wavelength grid [microns]
    spaxels : numpy.ndarray
        Pixel spectra (2d)

    Example
    -------
    >>> lam, spaxels = prep_spectral_hotspot_map()
    """

    # Start with toy map 2, which has modeled FpFs's for HD189
    lamhr, spaxin = prep_map2()

    # Parse into the 4 different FpFs
    FpFsA = spaxin[0,:]
    FpFsB = spaxin[100,:]
    FpFsC = spaxin[150,:]
    FpFsD = spaxin[450,:]

    # Outside hotspot
    FpFs1 = FpFsD

    # Inside hotspot
    FpFs2 = FpFsC

    # Get number of pixels/spaxels
    Npix = hp.nside2npix(Nside)

    # Calc the latitude and longitude of each hpix
    thetas, phis = hp.pix2ang(Nside, np.arange(Npix))
    thetas = thetas - np.pi / 2
    phis = phis - np.pi

    # Define empty 2d array for spaxels
    Nlam = len(lamhr)
    spaxels = np.zeros((Npix, Nlam))

    # Loop over pixels filling with blackbody
    for i in range(hp.nside2npix(Nside)):

        # Relabel variables for below equation
        phi1, theta1 =  thetas[i], phis[i]
        dtheta = theta0 - theta1

        # Calculate the angular distance between two points on sphere
        ang = np.arccos(np.sin(phi1) * np.sin(phi0) + np.cos(phi1) * np.cos(phi0) * np.cos(dtheta)) * 180 / np.pi

        # Use a different spectrum inside and outside of the great circle defined by dang
        if ang > ds:
            # Outside hotspot
            spaxels[i,:] = FpFs1
        else:
            # Within hotspot
            spaxels[i,:] = FpFs2

    return lamhr, spaxels



def create_quadrant_map(f1, f2, f3, f4, lat0 = 0.0, lon0 = 0.0, Nside = 16):
    """
    Return a quadrant map with different spectra 'painted on the surface'.

    Parameters
    ----------
    f1 : array
        Planet-star flux ratio 1 (upper/north left/west)
    f2 : array
        Planet-star flux ratio 2 (lower/south left/west)
    f3 : array
        Planet-star flux ratio 3 (upper/north right/east)
    f4 : array
        Planet-star flux ratio 4 (lower/south right/east)
    lat0 : float
        Latitude demarcation between north and south
    lon0 : float
        Longitude demarcation between east and west
    Nside : int
        Healpix NSIDE paramater

    Returns
    -------
    spaxels : numpy.ndarray
        Two-dimensional array of pixels by wavelengths

    """

    # Fluxes must be on same wavelength grid
    assert len(f1) == len(f2) == len(f3) == len(f4)

    # Get number of pixels/spaxels
    Npix = hp.nside2npix(Nside)

    # Calc the latitude and longitude of each hpix
    thetas, phis = hp.pix2ang(Nside, np.arange(Npix), lonlat=True)
    # Latitudes are the phis
    lat = phis
    # Convert to west (0 to -180) and east (0 to +180) longitudes
    lon = (thetas - 180.0)

    # Define empty 2d array for spaxels
    Nlam = len(f1)
    spaxels = np.zeros((Npix, Nlam))

    # Loop over pixels filling with spectra
    for i in range(Npix):

        # Latitudes north of demarcation
        if (lat[i] > lat0):

            # Longitudes west of demarcation
            if (lon[i] < lon0):

                # Set flux 1
                spaxels[i,:] = f1

            # Longitudes east of demarcation
            else:

                # Set flux 2
                spaxels[i,:] = f2

        # Latitudes south of demarcation
        else:

            # Longitudes west of demarcation
            if (lon[i] < lon0):

                # Set flux 3
                spaxels[i,:] = f3

            # Longitudes east of demarcation
            else:

                # Set flux 4
                spaxels[i,:] = f4

    return spaxels

def spec_flat_with_gaussian(lamhr, A = 0.1, B = 0.1, std = 0.01, xroll = 0.0):
    """
    Create a toy flat spectrum with a Gaussian spike.

    Parameters
    ----------
    lamhr : array
        Wavelength array
    A : float
        Flat line continuum value
    B : float
        Amplitude of Gaussian spike (added to continuum)
    std : float
        Standard deviation of Gaussian relative to the wavelength range
    xroll : float
        Fractional amount to shift location of Gaussian spike left (negative)
        or right (positive) from the central wavelength relative to the
        wavelength bounds (e.g. `xroll = 0.0` places Gaussian spike at the center,
        while `xroll = -0.25` shifts the spike halfway to the left bound).

    Returns
    -------
    FpFs : array
        Mock planet-to-star flux ratio with flat continuum and Gaussian spike
    """

    # Flat continuum
    y = A*np.ones(len(lamhr))

    # Gaussian spike
    g = signal.gaussian(len(lamhr), std*len(lamhr))
    g = np.roll(g, int(xroll*len(lamhr)))

    # Add spike to continuum
    FpFs = y + B*g

    return FpFs

def create_lightcurves_with_starry(lam, spaxels, lammin = 2.41, lammax = 3.98,
                                   dlam = 0.18, lmax = 18,
                                   plot_input_hp_maps = False,
                                   plot_lightcurves = True,
                                   plot_points_on_map_spec = False,
                                   points_on_map = [(-0.4, 0.4), (-0.4, -0.4), (0.4, 0.4), (0.4, -0.4), (0.0, 0.0)],
                                   plot_diagnostic = True,
                                   save_output = False,
                                   save_tag = "eclipse_lightcurve_test2"):
    """
    Creates toy multi-wavelength secondary eclipse lightcurves of HD189733 b

    Parameters
    ----------
    lam : numpy.array
        High-res model wavelength grid [microns]
    spaxels : numpy.ndarray
        2D array of HealPix number vs wavelength
    lammin : float
        Minimum wavelength
    lammax : float
        Maximum wavelength
    dlam : float
        Wavelength bin width
    lmax : int
        Largest spherical harmonic degree in the surface map
    plot_input_hp_maps : bool
        Plot the input HEALPix map
    plot_lightcurves : bool
        Plot the output lightcurves computed by `starry`
    plot_points_on_map_spec : bool
        Plot points on the visible map with the corresponding spectrum at
        each point
    points_on_map : list of (x,y) tuple
        Points to plot on map if `plot_points_on_map_spec` is set to True. Note
        that (0,0) is the center of the planet at secondary eclipse with the
        radius normalized to unity.
    plot_diagnostic : bool
        Make a diagnostic plot of many random spectra drawn from the visible
        disk of the planet to compare against the input spectra. This shows
        roughly how information is lost transforming the input spectral map to
        spherical harmonics.
    save_output : bool
        Set to save the output to a npz
    save_tag : str
        Name of file to save output to 

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

    # Construct low res wavelength grid
    lamlo, dlamlo = construct_lam(lammin, lammax, dlam=dlam)
    Nlamlo = len(lamlo)

    # Set HealPy pixel numbers
    Npix = spaxels.shape[0]

    # Define empty 2d array for spaxels
    spec2d = np.zeros((Npix, Nlamlo))

    # Loop over pixels filling with spectra
    for i in range(Npix):

        # Degrade the spectra to lower resolution
        spec2d[i,:] = downbin_spec(spaxels[i, :], lam, lamlo, dlam = dlamlo)

    # Make diagnostic plots
    if plot_input_hp_maps:
        indices = [0,5,-1]
        for i in indices:
            hp.mollview(spec2d[:,i], title=r"%0.2f $\mu$m" %lamlo[i])
            plt.show()

    # Create starry planet
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

    """
    # Rodrigo Luger's fix for starry spectral map normalization:
    """
    # Loop over wavelengths loading map into starry
    y = np.zeros_like(planet.y)
    for i in range(Nlamlo):
        # Expand a Healpix ring-ordered map in spherical harmonics
        # up to degree `lmax`
        y[:, i] = expand_hp(spec2d[:, i], lmax=lmax)
    # Renormalize planet map and luminosity accordingly
    planet[:, :] = y / y[0]
    planet.L = y[0]

    # Get the map intensity across the planet
    nx, ny = 300, 300
    x0 = np.linspace(-1, 1, nx)
    y0 = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x0, y0)
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

    # Apply vertical shift so the bottom of the eclipse is at 0.0
    lightcurve = lightcurve + (1.0 - np.max(lightcurve))

    if plot_lightcurves:
        # Make Plot
        fig, ax = plt.subplots(1, figsize=(14, 5))
        ax.set_xlabel('Time [days]')
        ax.set_ylabel('Relative Flux')
        for i in range(Nlamlo):
            # Subtract off minimum (which is the bottom of the eclipse) to look
            # like traditional form
            lc = lightcurve[:,i] - np.min(lightcurve[:,i])
            ax.plot(time, lc, c = "C%i" %(i%9), label = r"%.2f $\mu$m" %(lamlo[i]))
        ax.legend(ncol = 2)

    if plot_points_on_map_spec:

        # Which wavelength to plot map at
        iwavelength = 0

        fig, ax = plt.subplots(1, 2, figsize = (10,4))
        ax[1].set_xlabel(r"Wavelength [$\mu$m]")
        ax[1].set_ylabel(r"Relative Flux")
        ax[0].imshow(I[:,:,iwavelength],origin="lower")
        ax[0].axis('off')

        # Loop over points to plot
        for i in range(len(points_on_map)):

            # Get ith point
            xi, yi = points_on_map[i]

            # Scale point to array indices
            ix = int(np.round(0.5*nx*(xi + 1.0)))
            iy = int(np.round(0.5*ny*(yi + 1.0)))

            # Plot point on map
            ax[0].plot(ix,iy, "o", c="C%i" %(i%10), ms = 10)

            # Plot spectrum at point
            ax[1].plot(lamlo, I[iy,ix,:], c="C%i" %(i%10))

        plt.show()

    if plot_diagnostic:

        # Create figure
        fig, ax = plt.subplots(figsize = (12, 4))
        ax.set_ylabel("Fp/Fs")
        ax.set_xlabel("Wavelength [$\mu$m]")

        draws = 100
        # Loop over number of draws
        for k in range(draws):
            # Draw a random intensity from the planet
            i = np.random.randint(0,high=nx)
            j = np.random.randint(0,high=nx)
            # Plot it
            if k == 0:
                ax.plot(lamlo, I[i,j,:] * planet.L, color = "C1", lw = 0.5,
                        label = "starry samples", zorder=-1)
            else:
                ax.plot(lamlo, I[i,j,:] * planet.L, color = "C1", lw = 0.5,
                        zorder=-1)

        # Loop over input spaxels plotting each spectrum (creates thick lines as
        # they overplot)
        for i in range(Npix):
            if i == 0:
                ax.plot(lamlo, spec2d[i,:], color = "k", label = "Inputs")
            else:
                ax.plot(lamlo, spec2d[i,:], color = "k")

        ax.legend()

        plt.show()

    if save_output:
        pre = "data/input_lightcurves"
        fname = os.path.join(pre, save_tag+".npz")
        # Save generated lightcurve
        np.savez(fname, time = time, wl = lamlo,
                                    dwl = dlamlo, lightcurve = lightcurve)


    return time, lamlo, dlamlo, lightcurve

if __name__ == "__main__":

    # Get wavelength-dependent healpix map
    lamhr, spaxels = prep_spectral_hotspot_map()

    # Create mock lightcurves
    time, lam, dlam, lcurves = create_lightcurves_with_starry(lamhr, spaxels)
