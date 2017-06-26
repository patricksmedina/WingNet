# setup.py
#
# Compiles Cython code

from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("wingnet.heatmap.normalize_probs",
                ["wingnet/source/normalize_probs.pyx"],
                include_dirs = ['.',get_include()])

setup(ext_modules = cythonize(ext))
