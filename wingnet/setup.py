from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("heatmap.constructer",
                ["heatmap.constructer.pyx"],
                include_dirs = ['.',get_include()])
setup(ext_modules = cythonize(ext))
