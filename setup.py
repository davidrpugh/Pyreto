import os

from distutils.core import setup


def read(*paths):
    """Build a file path from *paths* and return the contents."""
    with open(os.path.join(*paths), 'r') as f:
        return f.read()

# Meta information
DESCRIPTION = "Python package for fitting Pareto distributions to data."

CLASSIFIERS = ['Development Status :: 3 - Alpha',
               'Intended Audience :: Education',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT License',
               'Operating System :: OS Independent',
               'Programming Language :: Python',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.3',
               'Programming Language :: Python :: 3.4',
               'Programming Language :: Python :: 3.5',
               'Topic :: Scientific/Engineering',
               ]

PACKAGES = ['pyreto', 'pyreto.testing']

setup(
    name="Pyreto",
    packages=PACKAGES,
    version='0.1.0a0',
    description=DESCRIPTION,
    long_description=read('README.rst'),
    license="MIT License",
    author="davidrpugh",
    author_email="david.pugh@maths.ox.ac.uk",
    url='https://github.com/davidrpugh/pyreto',
    classifiers=CLASSIFIERS,
    )
