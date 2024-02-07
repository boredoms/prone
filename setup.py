from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy 

extensions = [
    Extension("*", ["*.pyx"],
        include_dirs=[numpy.get_include()]),
]

setup(
    name="PRONE, a k-means clustering algorithm",
    version = "0.1.0",
    install_requires = [ "numpy", "cython" ],
    ext_modules=cythonize(extensions),
)