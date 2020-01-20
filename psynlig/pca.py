# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module defining plots for PCA results."""
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .colors import generate_colors


def pca_explained_variance(pca, axi=None, **kwargs):
    """Plot the explained variance as function of PCA components.

    Parameters
    ----------
    pca : object like :py:class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    axi : object like :py:class:`matplotlib.axes.Axes`, optional
        If given, the plot will be added to the specified axis.
        Otherwise, a new axis (and figure) will be created here.
    kwargs : dict
        Additional settings for plotting explained variance.

    Returns
    -------
    fig : object like :py:class:`matplotlib.figure.Figure`
        The figure containing the plot, if the figure is created
        here. Oterwise, it is None.
    axi : object like :py:class:`matplotlib.axes.Axes`
        The axis containing the plot.

    """
    fig = None
    if axi is None:
        fig, axi = plt.subplots(nrows=1, ncols=1)
    var = [0] + list(np.cumsum(pca.explained_variance_ratio_))
    comp = range(0, len(var))
    axi.plot(comp, var, **kwargs)
    axi.axhline(y=1, color='black', ls=':')
    axi.set(xlabel='Number of components', ylabel='Explained variance')
    axi.xaxis.set_major_locator(MaxNLocator(integer=True))
    if fig is not None:
        fig.tight_layout()
    return fig, axi


def pca_explained_variance_bar(pca, axi=None, **kwargs):
    """Plot the explained variance per principal component.

    Parameters
    ----------
    pca : object like :py:class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    axi : object like :py:class:`matplotlib.axes.Axes`, optional
        If given, the plot will be added to the specified axis.
        Otherwise, a new axis (and figure) will be created here.
    kwargs : dict
        Additional settings for plotting explained variance.

    Returns
    -------
    fig : object like :py:class:`matplotlib.figure.Figure`
        The figure containing the plot, if the figure is created
        here. Oterwise, it is None.
    axi : object like :py:class:`matplotlib.axes.Axes`
        The axis containing the plot.

    """
    fig = None
    if axi is None:
        fig, axi = plt.subplots(nrows=1, ncols=1)
    var = pca.explained_variance_ratio_
    comp = ['PC{}'.format(i + 1) for i in range(len(var))]
    xpos = range(len(var))
    axi.bar(xpos, var, **kwargs)
    axi.set_xticks(xpos)
    axi.set_xticklabels(
        comp,
        rotation='vertical',
    )
    axi.set(
        xlabel='Principal component',
        ylabel='Explained variance per component',
    )
    if fig is not None:
        fig.tight_layout()
    return fig, axi


def pca_explained_variance_pie(pca, axi=None, tol=1.0e-3):
    """Show the explained variance as function of PCA components.

    Parameters
    ----------
    pca : object like :py:class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    axi : object like :py:class:`matplotlib.axes.Axes`, optional
        If given, the plot will be added to the specified axis.
        Otherwise, a new axis (and figure) will be created here.
    tol : float, optional
        A tolerance for the missing variance. If the unexplained
        variance is less than this tolerance, it will not be shown.

    Returns
    -------
    fig : object like :py:class:`matplotlib.figure.Figure`
        The figure containing the plot, if the figure is created
        here. Oterwise, it is None.
    axi : object like :py:class:`matplotlib.axes.Axes`
        The axis containing the plot.

    """
    fig = None
    if axi is None:
        fig, axi = plt.subplots(nrows=1, ncols=1)
    var = list(pca.explained_variance_ratio_)
    missing = 1 - sum(var)
    comp = ['PC{}'.format(i + 1) for i in range(len(var))]
    if missing > tol:
        comp.append('Not explained')
        var.append(missing)
    colors = generate_colors(len(comp))
    axi.pie(
        var,
        labels=comp,
        colors=colors[:len(comp)],
        wedgeprops=dict(width=0.5, edgecolor='w'),
        textprops={'fontsize': 'x-large'},
    )
    axi.set(aspect='equal')
    if fig is not None:
        fig.tight_layout()
    return fig, axi
