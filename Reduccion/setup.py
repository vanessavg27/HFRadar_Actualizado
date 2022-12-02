# Copyright (c) 2012-2020 Jicamarca Radio Observatory
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
"""schainpy is an open source library to read, write and process radar data

Signal Chain is a radar data processing library wich includes modules to read,
and write different files formats, besides modules to process and visualize the
data.
"""

import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
#from schainpy import __version__

DOCLINES = __doc__.split("\n")
home = os.getenv("HOME")
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(

    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    ext_modules=[
        Extension("%s/Actualizado/Reduccion._noise"%(home), ["%s/Actualizado/Reduccion/_noise.c"%(home)]),
        ],
#    setup_requires = ["numpy"],
#    install_requires = [
#        "scipy",
#        "h5py",
#        "matplotlib",
#        "pyzmq",
#        "fuzzywuzzy",
#        "click",
#        ],
)
