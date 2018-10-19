# The input sph array will have the last two dimensions be (wavelengths,
# harmonic coefficients + 1). The first entry in each row is assumed to be
# the wavelength (in nm) corresponding to the spherical harmonic coefficients
# in that row. Note that the input array can have any dimensions prior to the
# last two.

# The returned array has dimensions (..., wavelengths, latitudes, longitudes),
# where the ellipses denote any extra dimensions from the input array.


def generate_maps(sph, N_lon, N_lat):
    import numpy as np
    from scipy.special import sph_harm

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
