from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy 

import os

currdir = os.getcwd()

setup(name="abnet",
      author="Rok Roskar",
      version='0.1.post1',
      author_email="roskar@ethz.ch",
      url="http://github.com/rokroskar/abnet",
      package_dir={'abnet/': ''},
      packages=['abnet'],
      ext_modules = cythonize([Extension("abnet.process_strings_cy", 
                               sources=["abnet/process_strings_cy.pyx"], 
                               include_dirs=[numpy.get_include()])]),
      scripts=['scripts/abnet-analyze'],
      install_requires=['click', 'findspark'],
      keywords=['pyspark','genomics', 'hpc','bioinformatics']
)
