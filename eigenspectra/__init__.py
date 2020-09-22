from __future__ import (division as _, print_function as _,
                absolute_import as _, unicode_literals as _)

# Version number
__version__ = "1.0.0"

# Was this imported from setup.py?
try:
    __EIGENSPECTRA_SETUP__
except NameError:
    __EIGENSPECTRA_SETUP__ = False

if not __EIGENSPECTRA_SETUP__:
    # This is a regular import
    from . import gen_lightcurves_new_starry
    from . import eigencurves_starry
    from . import eigenmaps
    from . import kmeans
    from . import bin_eigenspectra
    from . import run_higher_sph_harm
    from . import noise#eigensource
    from . import colormap2d
    from . import plot_utils
