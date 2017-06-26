from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("wingnet.heatmap.constructer",
                ["wingnet/source/constructer.pyx"],
                include_dirs = ['.',get_include()])

setup(ext_modules = cythonize(ext))
