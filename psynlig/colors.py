# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module defining helper methods for setting up colors."""
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap
import numpy as np


def generate_colors(ncolors, cmap=None):
    """Generate distinct colors.

    Parameters
    ----------
    ncolors : integer
        The number of colors to generate.
    cmap : string or object like :class:`matplotlib.colors.Colormap`, optional
        A specific color map to use. If not provided, a color map is
        selected based on the number of colors: If the number of colors
        is less than 10, we will use tab10; if the number is between 11
        and 20, we will use tab20; otherwise, we will use viridis.

    Returns
    -------
    out : list of objects like :class:`numpy.ndarray`
        The colors generated.

    """
    if cmap is not None:
        if isinstance(cmap, Colormap):
            return cmap(np.linspace(0, 1, ncolors))
        # Assume this is a valid colormap string then:
        cmap = get_cmap(name=cmap)
        return cmap(np.linspace(0, 1, ncolors))

    if ncolors <= 10:
        cmap = get_cmap(name='tab10')
        return cmap(np.linspace(0, 1, 10))

    if 10 < ncolors <= 20:
        cmap = get_cmap(name='tab20')
        return cmap(np.linspace(0, 1, 20))

    cmap = get_cmap(name='viridis')
    return cmap(np.linspace(0, 1, ncolors))


def generate_class_colors(class_data, cmap=None):
    """Generate colors for classes.

    Parameters
    ----------
    class_data : object like :class:`numpy.ndarray` or None
        The class labels for the data points. This is here assumed
        to be numerical values. If None are given we do not generate
        any colors here.
    cmap : string or object like :class:`matplotlib.colors.Colormap`, optional
        A color map to use for the classes.

    Returns
    -------
    color_class : list of objects like :class:`numpy.ndarray`
        A list containing the colors generated here for the classes.
    color_labels : dict of objects like :class:`numpy.ndarray`
        Colors for the different classes, that is ``color_labels[i]``
        contains the color for class ``i``.
    idx_class : dict of integers
        Indices for data classes. That is ``idx_class[i]`` contains
        the indices for the points in ``class_data`` which belongs
        to class ``i``.

    """
    classes = None
    color_class = None
    color_labels = {}
    idx_class = {}
    if class_data is not None:
        classes = list(set(class_data))
        color_class = generate_colors(len(classes), cmap=cmap)
        for i in classes:
            color_labels[i] = color_class[i]
            idx_class[i] = np.where(class_data == i)[0]
    return color_class, color_labels, idx_class
