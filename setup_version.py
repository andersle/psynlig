# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
psynlig -  A package for creating plots with matplotlib.

Copyright (C) 2020, Anders Lervik.
This file only generates the verison info.
"""
import ast
import os
import pathlib
import subprocess
import sys


VERSION_DEV = '{major:d}.{minor:d}.{micro:d}.dev{dev:d}'
VERSION = '{major:d}.{minor:d}.{micro:d}'

VERSION_FILE = pathlib.Path('psynlig').joinpath('version.py')

VERSION_TXT = '''# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""Version information for psynlig.

This file is generated by the script ``setup_version.py``.
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


def get_git_version():
    """Get the git revision as a string.

    Returns
    -------
    git_revision : string
        The git revision, it the git revision could not be determined,
        a 'unknown' will be returned.

    """
    git_revision = 'unknown'
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
        git_revision = 'unknown'
    return git_revision


def get_version_info(version):
    """Return the version number for psynlig.

    Parameters
    ----------
    version : dict
        A dict containing the current version info.

    Returns
    -------
    full_version : string
        The full version string for this release.
    git_revision : string
        The git revision number.

    """
    if pathlib.Path('.git').is_dir():
        git_revision = get_git_version()
    elif pathlib.Path(VERSION_FILE).is_file():
        try:
            from psynlig.version import git_revision
        except ImportError:
            raise ImportError('Unable to import git_revision. Try removing '
                              'psynlig/version.py and the build directory '
                              'before building.')
    else:
        git_revision = 'unknown'
    if not version['is_released']:
        git_version = ''.join(
            [
                version['version'].split('dev')[0],
                'dev{:d}+'.format(version['dev']),
                git_revision[:7]
            ]
        )
    else:
        git_version = version['version']
    full_version = version['version']
    return full_version, git_revision, git_version


def write_version_py(version):
    """Create a file with the version info for psynlig.

    Parameters
    ----------
    version : dict
        A dict containing the current version info.

    """
    with open(VERSION_FILE, 'wt') as vfile:
        vfile.write(
            VERSION_TXT.format(
                version=version['version'],
                full_version=version['full_version'],
                git_revision=version['git_revision'],
                git_version=version['git_version'],
                release=version['is_released'],
            )
        )
    return version['full_version']


def get_current_version():
    """Return the current major, minro, micro & dev version set."""
    version = {
        'major': None,
        'minor': None,
        'micro': None,
        'dev': None,
    }
    if pathlib.Path(VERSION_FILE).is_file():
        with open(VERSION_FILE, 'r') as infile:
            for lines in infile:
                if lines.startswith('FULL_VERSION ='):
                    version_line = ast.literal_eval(
                        lines.split('=')[1].strip()
                    )
                    split = version_line.split('.')
                    version['major'] = int(split[0])
                    version['minor'] = int(split[1])
                    version['micro'] = int(split[2])
                    if 'dev' in version_line:
                        version['dev'] = int(split[3].split('dev')[1])
    return version


def main(bump=False):
    """Set the version and create a version.py file.

    Parameters
    ----------
    bump : boolean, optional
        If bump is True, we will attempt to increase the micro version by 1.

    """
    # Hard-coded version:
    version = {
        'major': 0,
        'minor': 0,
        'micro': 1,
        'dev': 0,
        'is_released': True,
        'version': None,
        'git_revision': None,
        'git_version': None,
        'full_version': None
    }

    if bump:
        current = get_current_version()
        for key, val in current.items():
            if val is not None:
                val_new = val + 1 if key == 'micro' else val
                print('Setting {} to {}'.format(key, val_new))
                version[key] = val_new

    if version['is_released']:
        version_str = VERSION.format(
            major=version['major'],
            minor=version['minor'],
            micro=version['micro'],
        )
    else:
        version_str = VERSION_DEV.format(
            major=version['major'],
            minor=version['minor'],
            micro=version['micro'],
            dev=version['dev'],
        )
    version['version'] = version_str

    full_version, git_revision, git_version = get_version_info(version)

    version['full_version'] = full_version
    version['git_revision'] = git_revision
    version['git_version'] = git_version

    print(
        'Setting version to: {}'.format(
            write_version_py(version)
        )
    )


if __name__ == '__main__':
    main(bump=len(sys.argv)>1)