# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module for generating scatter plots of varialbes."""
from itertools import combinations
import warnings
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import comb
from .colors import generate_colors


_MAX_PLOTS = 5
_WARNING_MAX_PLOTS = (
    'This will generate {} plots. If you really want to generate '
    'all these plots, rerun the function with the '
    'argument "force=True".'
)


def _generate_class_colors(class_data):
    """Generate colors for classes.

    Parameters
    ----------
    class_data : object like :py:class:``, optional
        The labels for the data.

    Returns
    -------
    color_class : list of numpy.arrays
        The colors generated.
    color_labels : dict of numpy.arrays
        Colors for the different classes.
    idx_class : dict of integers
        Indexes for data classes.

    """
    classes = None
    color_class = None
    color_labels = {}
    idx_class = {}
    if class_data is not None:
        classes = list(set(class_data))
        color_class = generate_colors(len(classes))
        for i in classes:
            color_labels[i] = color_class[i]
            idx_class[i] = np.where(class_data == i)[0]
    return color_class, color_labels, idx_class


def _generate_scatter_legend(axi, color_labels, class_names, **kwargs):
    """Generate legend for a scatter plot."""
    patches, labels = [], []
    for key, val in color_labels.items():
        patches.append(
            axi.scatter([], [], color=val, **kwargs)
        )
        if class_names is not None:
            label = class_names.get(key, key)
        else:
            label = key
        labels.append(label)
    return patches, labels


def plot_scatter(data, xvar, yvar, axi=None, class_data=None, class_names=None,
                 **kwargs):
    """Make a 2D scatter plot of the given data.

    Parameters
    ----------
    data : object like :py:class:`pandas.DataFrame`
        The data we are plotting.
    xvar : string
        The column to use as the x-variable.
    yvar : string
        The column to use as the y-variable.
    axi : object like :py:class:``, optional
        An axes to add the plot to. If this is not provided,
        a new axis (and figure) will be created here
    class_data : object like, optional
        Class information for the points (if available).
    class_names : dict of strings
        A mapping from the class data to labels/names.

    """
    color_class, color_labels, idx_class = _generate_class_colors(class_data)
    fig = None
    if axi is None:
        fig, axi = plt.subplots()
    axi.set(xlabel=xvar, ylabel=yvar)

    if class_data is None:
        axi.scatter(data[xvar], data[yvar], **kwargs)
    else:
        for class_id, idx in idx_class.items():
            axi.scatter(
                data[xvar][idx],
                data[yvar][idx],
                color=color_class[class_id],
                **kwargs
            )
        patches, labels = _generate_scatter_legend(
            axi, color_labels, class_names, **kwargs
        )
        axi.legend(patches, labels)
    if fig is not None:
        fig.tight_layout()
    return fig, axi


def plot_3d_scatter(data, xvar, yvar, zvar, class_data=None, class_names=None,
                    **kwargs):
    """Make a 3D scatter plot of the given data.

    Parameters
    ----------
    data : object like :py:class:`pandas.DataFrame`
        The data we are plotting.
    xvar : string
        The column to use as the x-variable.
    yvar : string
        The column to use as the y-variable.
    zvar : string
        The column to use as the z-variable
    class_data : object like, optional
        Class information for the points (if available).
    class_names : dict of strings
        A mapping from the class data to labels/names.

    """
    color_class, color_labels, idx_class = _generate_class_colors(class_data)
    fig = plt.figure()
    axi = fig.add_subplot(111, projection='3d')
    axi.set_xlabel(xvar, labelpad=15)
    axi.set_ylabel(yvar, labelpad=15)
    axi.set_zlabel(zvar, labelpad=15)

    if class_data is None:
        axi.scatter(data[xvar], data[yvar], data[zvar], **kwargs)
    else:
        for class_id, idx in idx_class.items():
            axi.scatter(
                data[xvar][idx],
                data[yvar][idx],
                data[zvar][idx],
                color=color_class[class_id],
                **kwargs
            )
        patches, labels = _generate_scatter_legend(
            axi, color_labels, class_names, **kwargs
        )
        axi.legend(patches, labels)
    fig.tight_layout()
    return fig, axi


def generate_3d_scatter(data, variables, class_data=None, class_names=None,
                        force=False, **kwargs):
    """Generate 3D scatter plots from the given data and variables."""
    figures = []
    axes = []
    if len(variables) < 3:
        raise ValueError(
            'For generating 3D plots, at least 3 variables must be provided.'
        )
    nplots = comb(len(variables), 3, exact=True)
    if nplots > _MAX_PLOTS and not force:
        msg = _WARNING_MAX_PLOTS.format(nplots)
        warnings.warn(msg)
        return figures, axes
    for (xvar, yvar, zvar) in combinations(variables, 3):
        figi, axi = plot_3d_scatter(
            data,
            xvar,
            yvar,
            zvar,
            class_data=class_data,
            class_names=class_names,
            **kwargs
        )
        figures.append(figi)
        axes.append(axi)
    return figures, axes
