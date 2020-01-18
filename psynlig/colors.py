# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module defining helper metods for setting up colors."""
from matplotlib.cm import get_cmap
import numpy as np


def generate_colors(ncolors, cmap_name=None):
    """Generate distinct colors.

    Parameters
    ----------
    ncolors : int
        The number of colors to generate.
    cmap_name : string, optional
        A specific color map to use. If not provided, a color map
        is selected based on the number of colors: If the number
        of colors is less than 10, we will use tab10; if the number
        is between 11 and 20, we will use tab20; otherwise, we will
        use viridis.

    Returns
    -------
    out : list of numpy.arrays
        The colors generated.

    """
    if cmap_name is not None:
        cmap = get_cmap(name=cmap_name)
        return cmap(np.linspace(0, 1, ncolors))
    if ncolors <= 10:
        cmap = get_cmap(name='tab10')
        return cmap(np.linspace(0, 1, 10))
    if 10 < ncolors <= 20:
        cmap = get_cmap(name='tab20')
        return cmap(np.linspace(0, 1, 20))
    cmap = get_cmap(name='viridis')
    return cmap(np.linspace(0, 1, ncolors))
