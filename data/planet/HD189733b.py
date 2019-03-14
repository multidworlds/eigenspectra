###################################################################
#   HD 189733b SYSTEM PROPERTIES
###################################################################

import astropy.constants as C
import astropy.units as U

# The published properties of the system. The Astropy unit conversions will
# automatically convert units to the appropriate ones for input to SPIDERMAN.

# Orbital period
orbital_period = 2.21857567 * U.d

# Absolute value of the semi-major axis
semimajor_axis = 0.0313 * U.AU

# Orbital inclination
orbital_inclination = 85.71 * U.deg

# Orbital eccentricity
orbital_eccentricity = 0.0

# Argument of periastron (taken to be 90 degrees if ecc = 0,
# so that transit is coincident with periastron)
argument_of_periastron = 90 * U.deg

# Planet radius
planet_radius = 1.216 * C.R_jup

# Stellar radius
stellar_radius = 0.805 * C.R_sun

# Planetary limb-darkening parameters (u1, u2)
planet_limbdarkening = [0, 0]

properties = {

    "per": orbital_period.to(U.d).value,
    "a_abs": semimajor_axis.to(U.AU).value,
    "inc": orbital_inclination.to(U.deg).value,
    "ecc": orbital_eccentricity,
    "w": argument_of_periastron.to(U.deg).value,
    # Planet to star radius ratio
    "rp": (planet_radius/stellar_radius).decompose().value,
    # Semi-major axis to stellar radius ratio
    "a": (semimajor_axis/stellar_radius).decompose().value,
    "p_u1": planet_limbdarkening[0],
    "p_u2": planet_limbdarkening[1],

}

# (PYNRC ONLY) For error generation with pynrc, provide the spectral type,
# magnitude, including Astropy-compatible unit (default "vegamag"), and the
# corresponding bandpass (using pysynphot syntax).

# Spectral type
spectral_type = "K0V"

# Magnitude, including units and bandpass.
stellar_magnitude = ["5.541", "vegamag"]
stellar_bandpass = "johnson,k"
