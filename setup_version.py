# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
psynlig -  A package for creating plots with matplotlib.

Copyright (C) 2020, Anders Lervik.
This file generates the version info.
"""
import argparse
import json
import os
import pathlib
import subprocess


# Path to file containing current version info:
CURRENT_VERSION_FILE = pathlib.Path('version.json')
# Path to file with version info accessible by psynlig:
VERSION_FILE = pathlib.Path('psynlig').joinpath('version.py')
# Path to the setup.py file:
SETUP_PY = pathlib.Path('setup.py')
# Format for versions:
VERSION_DEV_FMT = '{major:d}.{minor:d}.{micro:d}.dev{dev:d}'
VERSION_FMT = '{major:d}.{minor:d}.{micro:d}'
# Format for version.py
VERSION_TXT = '''# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""Version information for psynlig.

This file is generated by psynlig (``setup_version.py``).
"""
SHORT_VERSION = '{version:s}'
VERSION = '{version:s}'
FULL_VERSION = '{full_version:s}'
GIT_REVISION = '{git_revision:s}'
GIT_VERSION = '{git_version:s}'
RELEASE = {release:}

if not RELEASE:
    VERSION = GIT_VERSION
'''


def generate_version_string(version):
    """Generate a string with the current pysynlig version.

    Parameters
    ----------
    version : dict
        A dict containing the current psynlig version.

    Returns
    -------
    version_txt : string
        A string with the current psynlig version.

    """
    version_fmt = VERSION_FMT if version['release'] else VERSION_DEV_FMT
    return version_fmt.format(**version)


def get_git_version():
    """Obtain the git revision as a string.

    This method is adapted from Numpy's setup.py

    Returns
    -------
    git_revision : string
        The git revision, it the git revision could not be determined,
        a 'Unknown' will be returned.

    """
    git_revision = 'Unknown'
    try:
        env = {}
        for key in ('SYSTEMROOT', 'PATH'):
            val = os.environ.get(key)
            if val is not None:
                env[key] = val
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(['git', 'rev-parse', 'HEAD'],
                               stdout=subprocess.PIPE,
                               env=env).communicate()[0]
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = 'Unknown'
    return git_revision


def get_version_info(version):
    """Return the version number for psynlig.

    This method is adapted from Numpy's setup.py.

    Parameters
    ----------
    version : dict
        The current version information.

    Returns
    -------
    full_version : string
        The full version string for this release.
    git_revision : string
        The git revision number.

    """
    version_txt = generate_version_string(version)
    if pathlib.Path('.git').is_dir():
        git_revision = get_git_version()
    elif pathlib.Path(VERSION_FILE).is_file():
        try:
            from psynlig.version import git_revision
        except ImportError:
            raise ImportError(
                'Unable to import git_revision. Try removing '
                'psynlig/version.py and the build directory '
                'before building.'
            )
    else:
        git_revision = 'Unknown'
    if not version['release']:
        git_version = ''.join(
            [
                version_txt.split('dev')[0],
                'dev{:d}+'.format(version['dev']),
                git_revision[:7]
            ]
        )
    else:
        git_version = version_txt
    full_version = version_txt
    return full_version, git_revision, git_version


def write_version_py(version):
    """Create a file with the version info for psynlig.

    This method is adapted from Numpy's setup.py.

    Parameters
    ----------
    version : dict
        The dict containing the current version information.

    Returns
    -------
    full_version : string
        The current full version for psynlig. 

    """
    full_version, git_revision, git_version = get_version_info(version)
    version_txt = VERSION_TXT.format(
        version=full_version,
        full_version=full_version,
        git_revision=git_revision,
        git_version=git_version,
        release=version['release'],
    )
    with open(VERSION_FILE, 'wt') as vfile:
        vfile.write(version_txt)
    return full_version


def write_version_in_setup_py(version):
    """Update version for setup.py."""
    tmp = []
    comment = '# Automatically set by setup_version.py'
    with open(SETUP_PY, 'r') as sfile:
        for lines in sfile:
            if lines.startswith('FULL_VERSION ='):
                tmp.append(
                    ("FULL_VERSION = '{}'  {}\n".format(version, comment))
                )
            else:
                tmp.append(lines)
    with open(SETUP_PY, 'wt') as sfile:
        for lines in tmp:
            sfile.write(lines)


def bump_version(args, version):
    """Increment the version number if requested.

    Parameters
    ----------
    args : object like argparse.Namespace
        The arguments determining if we are to bump the version number.
    version : dict
        The current version.

    Returns
    -------
    new_version : dict
        The updated version (if an update is requested). Otherwise it
        is just a copy of the input version.

    """
    new_version = version.copy()
    if args.bump_dev:
        new_version['dev'] += 1
    if args.bump_micro:
        new_version['micro'] += 1
        new_version['dev'] = 0
    if args.bump_minor:
        new_version['minor'] += 1
        new_version['micro'] = 0
        new_version['dev'] = 0
    if args.bump_major:
        new_version['major'] += 1
        new_version['minor'] = 0
        new_version['micro'] = 0
        new_version['dev'] = 0
    return new_version


def main(args):
    """Generate version information and update the relevant files."""
    version = {}
    with open(CURRENT_VERSION_FILE, 'r') as json_file:
        version = json.load(json_file)
    version = bump_version(args, version)
    full_version = write_version_py(version)
    print('Setting version to: {}'.format(full_version))
    write_version_in_setup_py(full_version)
    with open(CURRENT_VERSION_FILE, 'w') as json_file:
        json.dump(version, json_file, indent=4)


def get_argument_parser():
    """Return a parser for arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bump_major',
        action='store_true',
        help='Increment the major version.'
    )
    parser.add_argument(
        '--bump_minor',
        action='store_true',
        help='Increment the minor version.'
    )
    parser.add_argument(
        '--bump_micro',
        action='store_true',
        help='Increment the micro version.'
    )
    parser.add_argument(
        '--bump_dev',
        action='store_true',
        help='Increment the development version.'
    )
    return parser


if __name__ == '__main__':
    PARSER = get_argument_parser()
    main(PARSER.parse_args())
