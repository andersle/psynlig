# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module defining plots for PCA results."""
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .colors import generate_colors


def pca_explained_variance(pca, **kwargs):
    """Plot the explained variance as function of PCA components.

    Parameters
    ----------
    pca : object like :py:class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    kwargs : dict
        Additional settings for plotting explained variance.

    Returns
    -------
    fig : object like :py:class:`matplotlib.figure.Figure`
        The figure containing the plot.
    axes : list of objects like :py:class:`matplotlib.axes.Axes`
        The axes containing the plots.

    """
    fig, axes = plt.subplots(nrows=1, ncols=2)
    var = [0] + list(np.cumsum(pca.explained_variance_ratio_))
    comp = range(0, len(var))
    axes[1].plot(comp, var, **kwargs)
    axes[1].axhline(y=1, color='black', ls=':')
    axes[1].set(xlabel='Number of components', ylabel='Explained variance')
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    var = pca.explained_variance_ratio_
    comp = ['PC{}'.format(i + 1) for i in range(len(var))]
    xpos = range(len(var))
    axes[0].bar(xpos, var)
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels(
        comp,
        rotation='vertical',
    )
    axes[0].set(
        xlabel='Principal component',
        ylabel='Explained variance per component',
    )
    fig.tight_layout()
    return fig, axes


def pca_explained_variance_pie(pca, tol=1.0e-3):
    """Show the explained variance as function of PCA components.

    Parameters
    ----------
    pca : object like :py:class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    tol : float, optional
        A tolerance for the missing variance. If the unexplained
        variance is less than this tolerance, it will not be shown.

    Returns
    -------
    fig : object like :py:class:`matplotlib.figure.Figure`
        The figure containing the plot.
    axi : object like :py:class:`matplotlib.axes.Axes`
        The axis containing the plots.

    """
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
    axi.set_title('Explained variance by principal components',
                  fontsize='xx-large')
    fig.tight_layout()
    return fig, axi
