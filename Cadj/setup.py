from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("adj_set_nettorkx.pyx")
)