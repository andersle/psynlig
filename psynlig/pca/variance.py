# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module defining plots for PCA variance."""
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from psynlig.colors import generate_colors


def _create_figure_if_needed(axi, figsize=None):
    """Create a figure if needed (axi is None)."""
    fig = None
    if axi is None:
        if figsize is None:
            fig, axi = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
        else:
            fig, axi = plt.subplots(
                figsize=figsize, nrows=1, ncols=1, constrained_layout=True
            )
    return fig, axi


def pca_explained_variance(pca, axi=None, figsize=None, **kwargs):
    """Plot the explained variance as function of PCA components.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    axi : object like :class:`matplotlib.axes.Axes`, optional
        If given, the plot will be added to the specified axis.
        Otherwise, a new axis (and figure) will be created here.
    figsize : tuple of ints, optional
        A desired size of the figure, if created here.
    kwargs : dict, optional
        Additional settings for plotting explained variance.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure containing the plot, if the figure is created
        here. Oterwise, it is None.
    axi : object like :class:`matplotlib.axes.Axes`
        The axis containing the plot.

    """
    fig, axi = _create_figure_if_needed(axi, figsize=figsize)
    var = [0] + list(np.cumsum(pca.explained_variance_ratio_))
    comp = range(0, len(var))
    axi.plot(comp, var, **kwargs)
    axi.axhline(y=1, color='black', ls=':')
    axi.set(xlabel='Number of components',
            ylabel='Explained variance (fraction)')
    axi.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig, axi


def pca_residual_variance(pca, axi=None, figsize=None, **kwargs):
    """Plot the residual variance as function of PCA components.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    axi : object like :class:`matplotlib.axes.Axes`, optional
        If given, the plot will be added to the specified axis.
        Otherwise, a new axis (and figure) will be created here.
    figsize : tuple of ints, optional
        A desired size of the figure, if created here.
    kwargs : dict, optional
        Additional settings for plotting explained variance.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure containing the plot, if the figure is created
        here. Oterwise, it is None.
    axi : object like :class:`matplotlib.axes.Axes`
        The axis containing the plot.

    """
    fig, axi = _create_figure_if_needed(axi, figsize=figsize)
    var = 1 - np.array([0] + list(np.cumsum(pca.explained_variance_ratio_)))
    comp = range(0, len(var))
    axi.axhline(y=0, color='black', ls=':')
    axi.plot(comp, var, **kwargs)
    axi.set(xlabel='Number of components',
            ylabel='Residual variance (fraction)')
    axi.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig, axi


def pca_scree(pca, axi=None, figsize=None, **kwargs):
    """Plot the eigenvalues as function of PCA components.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    axi : object like :class:`matplotlib.axes.Axes`, optional
        If given, the plot will be added to the specified axis.
        Otherwise, a new axis (and figure) will be created here.
    figsize : tuple of ints, optional
        A desired size of the figure, if created here.
    kwargs : dict, optional
        Additional settings for the plotting.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure containing the plot, if the figure is created
        here. Oterwise, it is None.
    axi : object like :class:`matplotlib.axes.Axes`
        The axis containing the plot.

    """
    fig, axi = _create_figure_if_needed(axi, figsize=figsize)
    eigenvalues = pca.explained_variance_
    comp = range(1, len(eigenvalues) + 1)
    axi.plot(comp, eigenvalues, **kwargs)
    axi.set(xlabel='Principal component',
            ylabel='Eigenvalue')
    axi.xaxis.set_major_locator(MaxNLocator(integer=True))
    axi.set_xlim(min(comp) - 0.25, max(comp) + 0.25)
    return fig, axi


def pca_explained_variance_bar(pca, axi=None, figsize=None, **kwargs):
    """Plot the explained variance per principal component.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    axi : object like :class:`matplotlib.axes.Axes`, optional
        If given, the plot will be added to the specified axis.
        Otherwise, a new axis (and figure) will be created here.
    figsize : tuple of ints, optional
        A desired size of the figure, if created here.
    kwargs : dict, optional
        Additional settings for plotting explained variance.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure containing the plot, if the figure is created
        here. Oterwise, it is None.
    axi : object like :class:`matplotlib.axes.Axes`
        The axis containing the plot.

    """
    fig, axi = _create_figure_if_needed(axi, figsize=figsize)
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
        ylabel='Explained variance (fraction) per component',
    )
    return fig, axi


def pca_explained_variance_pie(pca, axi=None, figsize=None,
                               cmap=None, tol=1.0e-3):
    """Show the explained variance as function of PCA components in a pie.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    axi : object like :class:`matplotlib.axes.Axes`, optional
        If given, the plot will be added to the specified axis.
        Otherwise, a new axis (and figure) will be created here.
    figsize : tuple of ints, optional
        A desired size of the figure, if created here.
    cmap : string or object like :class:`matplotlib.colors.Colormap`, optional
        The color map to use for generating colors.
    tol : float, optional
        A tolerance for the missing variance. If the unexplained
        variance is less than this tolerance, it will not be shown in
        the pie chart.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure containing the plot, if the figure is created
        here. Oterwise, it is None.
    axi : object like :class:`matplotlib.axes.Axes`
        The axis containing the plot.

    """
    fig, axi = _create_figure_if_needed(axi, figsize=figsize)
    var = list(pca.explained_variance_ratio_)
    missing = 1 - sum(var)
    comp = ['PC{}'.format(i + 1) for i in range(len(var))]
    if missing > tol:
        comp.append('Not explained')
        var.append(missing)
    colors = generate_colors(len(comp), cmap=cmap)
    axi.pie(
        var,
        labels=comp,
        colors=colors[:len(comp)],
        wedgeprops=dict(width=0.5, edgecolor='w'),
        textprops={'fontsize': 'x-large'},
        normalize=False,
    )
    axi.set(aspect='equal')
    return fig, axi
