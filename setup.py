from __future__ import print_function
from setuptools import setup
from distutils.extension import Extension
import warnings
import os

use_cython = 1

try: 
    from Cython.Build import cythonize
except ImportError: 
    use_cython = 0
    warnings.warn('No cython found -- install will use pre-generated C files')
try: 
    import numpy     
    use_cython *= 1
except ImportError:
    use_cython = 0
    warnings.warn('No numpy found -- install will use pre-generated C files')

currdir = os.getcwd()

ext_modules = [Extension("imnet.process_strings_cy", 
                               sources=["imnet/process_strings_cy.pyx"], 
                               include_dirs=[numpy.get_include()])]

if use_cython: 
    print('Using cython')
    ext_modules = cythonize(ext_modules)


setup(name="imnet",
      author="Rok Roskar",
      version='0.1.post2',
      author_email="roskar@ethz.ch",
      url="http://github.com/rokroskar/imnet",
      package_dir={'imnet/': ''},
      packages=['imnet'],
      ext_modules = ext_modules,
      scripts=['scripts/imnet-analyze'],
      install_requires=['click', 'findspark', 'python-Levenshtein', 'scipy', 'networkx', 'pandas'],
      keywords=['pyspark','genomics', 'hpc','bioinformatics']
)
