# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
pysynlig - A package for creating plots with matplotlib.

Copyright (C) 2020, Anders Lervik.
"""
import ast
import pathlib
from setuptools import setup, find_packages


HERE = pathlib.Path(__file__).parent.resolve()


def get_long_description():
    """Return the contents of the README.md file."""
    # Get the long description from the README file
    long_description = ''
    readme = HERE.joinpath('README.md')
    with open(readme, 'r') as fileh:
        long_description = fileh.read()
    return long_description


def get_version():
    """Read the version from the version.py file."""
    filename = HERE.joinpath('psynlig', 'version.py')
    with open(filename, 'r') as fileh:
        for lines in fileh:
            if lines.startswith('FULL_VERSION ='):
                version = ast.literal_eval(lines.split('=')[1].strip())
                return version
    return 'unknown'


def get_requirements():
    """Read requirements from the requirements.txt file."""
    requirements = []
    filename = HERE.joinpath('requirements.txt')
    with open(filename, 'r') as fileh:
        for lines in fileh:
            requirements.append(lines.strip())
    return requirements


setup(
    name='psynlig',
    version=get_version(),
    description='A package for creating plots with matplotlib.',
    long_description=get_long_description(),
    url='https://github.com/andersle/psynlig',
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
        'Programming Language :: Python :: 3.7',
        'Topic :: Other/Nonlisted Topic',
    ],
    keywords='matplotlib',
    packages=find_packages(),
    install_requires=get_requirements(),
)
