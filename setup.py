from setuptools import setup
from distutils.extension import Extension
import warnings
import os

try: 
    from Cython.Build import cythonize
except ImportError: 
    warnings.warn('No cython found -- install will use pre-generated C files')
try: 
    import numpy     
except ImportError:
    warnings.warn('No numpy found -- install will use pre-generated C files')

currdir = os.getcwd()

setup(name="abnet",
      author="Rok Roskar",
      version='0.1.post3',
      author_email="roskar@ethz.ch",
      url="http://github.com/rokroskar/abnet",
      package_dir={'abnet/': ''},
      packages=['abnet'],
      ext_modules = cythonize([Extension("abnet.process_strings_cy", 
                               sources=["abnet/process_strings_cy.pyx"], 
                               include_dirs=[numpy.get_include()])]),
      scripts=['scripts/abnet-analyze'],
      install_requires=['click', 'findspark', 'python-Levenshtein', 'scipy', 'networkx', 'pandas'],
      keywords=['pyspark','genomics', 'hpc','bioinformatics']
)
