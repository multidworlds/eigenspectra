#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from setuptools import setup

# Hackishly inject a constant into builtins to enable importing of the
# module in "setup" mode. Borrowed from `kplr`
import sys
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__EIGENSPECTRA_SETUP__ = True
import eigenspectra

long_description = \
    """Insert long description"""

# Setup!
setup(name='eigenspectra',
      version=eigenspectra.__version__,
      description='Short Description.',
      long_description=long_description,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Astronomy',
      ],
      url='https://github.com/multidworlds/eigenspectra',
      author='MultiDWorlds',
      author_email='jlustigy@gmail.com',
      license = 'MIT',
      packages=['eigenspectra'],
      install_requires=[
                        'numpy',
                        'scipy',
                        'matplotlib',
                        'astropy',
                        'emcee',
                        'corner',
                        'healpy',
                        'starry==1.0.0',
                        'xarray==0.16.0', # TODO: This is a hack fix b/c 0.16.1 introduced a bug
                        'spiderman-package',
                        'colormath',
                        'colorcet',
                        'sphinx',
                        'sphinx-rtd-theme',
                        'nbsphinx', 
                        ],
      dependency_links=[],
      scripts=[],
      include_package_data=True,
      zip_safe=False,
      data_files=[]
      )
