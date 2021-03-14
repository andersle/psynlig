# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
pysynlig - A package for creating plots with matplotlib.

Copyright (C) 2020, Anders Lervik.
"""
import ast
import pathlib
from setuptools import setup, find_packages


GITHUB = 'https://github.com/andersle/psynlig'
DOCS = 'https://psynlig.readthedocs.io/en/latest'

FULL_VERSION = '0.2.1.dev0'  # Automatically set by setup_version.py

def get_long_description():
    """Hard-coded long description."""
    long_description = (
        'psynlig is a small package for generating '
        'plots using `matplotlib <https://www.matplotlib.org/>`_. '
        'It is intended as a library of plotting functions that can be '
        'used to streamline investigation of data sets.\n'
        'The psynlig documentation can be found at `{docs} <{docs}>`_'
        ' and the source code is hosted at `{github} <{github}>`_.'
    )
    return long_description.format(docs=DOCS, github=GITHUB)


def get_version():
    """Return the version from version.py as a string."""
    here = pathlib.Path(__file__).absolute().parent
    filename = here.joinpath('psynlig', 'version.py')
    with open(filename, 'r') as fileh:
        for lines in fileh:
            if lines.startswith('FULL_VERSION ='):
                version = ast.literal_eval(lines.split('=')[1].strip())
                return version
    return FULL_VERSION


def get_requirements():
    """Read requirements.txt and return a list of requirements."""
    here = pathlib.Path(__file__).absolute().parent
    requirements = []
    filename = here.joinpath('requirements.txt')
    with open(filename, 'r') as fileh:
        for lines in fileh:
            requirements.append(lines.strip())
    return requirements


setup(
    name='psynlig',
    version=get_version(),
    description='A package for creating plots with matplotlib.',
    long_description=get_long_description(),
    url=GITHUB,
    author='Anders Lervik',
    author_email='andersle@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Other/Nonlisted Topic',
    ],
    keywords='matplotlib',
    packages=find_packages(),
    install_requires=get_requirements(),
)
