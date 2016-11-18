from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy 

import os

currdir = os.getcwd()

setup(name="abnet",
      author="Rok Roskar",
      author_email="roskar@ethz.ch",
      package_dir={'abnet/': ''},
      packages=['abnet'],
      ext_modules = cythonize([Extension("abnet.process_strings_cy", 
                               sources=["abnet/process_strings_cy.pyx"], 
                               include_dirs=[numpy.get_include()])]),
      scripts=['scripts/abnet-analyze']
)
