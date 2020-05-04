import sys
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["Discriminator.pyx", "Utils.pyx"], compiler_directives={"language_level": sys.version_info[0]}, annotate=True)

)