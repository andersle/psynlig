# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""Insert code examples from files into README.md"""
import pathlib
import sys


def read_current_readme(filename):
    """Read the current readme file."""
    txt = []
    source_files = {}
    skip_code = False
    files = 0
    with open(filename, 'r') as infile:
        for lines in infile:
            if lines.startswith('```python'):
                filename = None
                try:
                    filename = lines.split('python')[1].strip()
                    source_files[files] = filename
                    files += 1
                except IndexError:
                    pass
                txt.append(lines)
                if filename is not None:
                    skip_code = True
                    txt.append('{{sourcefile_{}}}'.format(files - 1))
            else:
                if lines.strip() == '```':
                    skip_code = False
                if not skip_code:
                    txt.append(lines)
    return ''.join(txt), source_files


def read_source_file(filename, skiplines=3):
    """Read the source file, skip the number of given lines."""
    txt_list = []
    with open(filename, 'r') as infile:
        for i, lines in enumerate(infile):
            if i >= skiplines:
                txt_list.append(lines)
    return ''.join(txt_list)


def main(readme):
    """Read current readme and insert code."""
    # Read current README:
    txt, source_files = read_current_readme(readme)
    readme_dir = pathlib.Path(readme).resolve().parent
    txt_source = {}
    # Read source files:
    for key, val in source_files.items():
        filename = readme_dir.joinpath(val)
        txt_source['sourcefile_{}'.format(key)] = read_source_file(filename)
    # Create updated README:
    new_file = txt.format(**txt_source)
    print(new_file.strip())


if __name__ == '__main__':
    main(sys.argv[1])
