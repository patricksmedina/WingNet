from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("construct_heatmap", ["construct_heatmap.pyx"],
                include_dirs = ['.',get_include()])
setup(ext_modules = cythonize(ext))
