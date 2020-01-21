# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module defining plots for PCA results."""
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .colors import generate_colors
from .common import MARKERS


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


def pca_1d_loadings(pca, xvars, select_components=None):
    """Plot the loadings from a PCA in a 1D plot.

    Parameters
    ----------
    pca : object like :py:class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    xvars : list of strings
        Labels for the original variables.
    select_componets : set of integers, optional
        This variable can be used to select the principal components
        we will create plot for. Note that the principal component
        numbering will here start from 1 (and not 0). If this is not
        given, all will be plotted.

    Returns
    -------
    figures : list of objects like :py:class:`matplotlib.figure.Figure`
        The figures containing the plots.
    axes : list of objects like :py:class:`matplotlib.axes.Axes`
        The axes containing the plots.

    """
    figures = []
    axes = []
    components = pca.n_components_
    colors = generate_colors(len(xvars))
    for i in range(components):
        if select_components is None:
            pass
        else:
            if (i + 1) not in select_components:
                continue
        fig, axi = plt.subplots()
        axi.set_title('Principal component {}'.format(i + 1))
        coefficients = np.transpose(pca.components_[i, :])
        pca_1d_loadings_component(axi, coefficients, xvars, colors)
        fig.tight_layout()
        figures.append(fig)
        axes.append(axi)
    return figures, axes


def pca_1d_loadings_component(axi, coefficients, xvars, colors):
    """Plot the loadings for a single component in a 1D plot

    Parameters
    ----------
    axi : object like :py:class:`matplotlib.axes.Axes`
        The plot we will add the loadings to.
    coefficients : object like :py:class:`numpy.ndarray`
        The coefficients we are to show.
    xvars : list of strings
        Labels for the original variables.
    colors : list of floats or strings
        The colors used for the different labels.

    """
    pos_b, pos_t = 0, 0
    for j, coeff in enumerate(coefficients):
        # Add marker:
        axi.scatter(
            coeff,
            0,
            label=xvars[j],
            marker=MARKERS.get(j, 'o'),
            color=colors[j],
            zorder=4,
        )
        if j % 2 == 0:
            pos_b += 1
            ypos = -2 - pos_b
            valign = 'top'
        else:
            pos_t += 1
            ypos = 2 + pos_t
            valign = 'bottom'
        # Add text:
        axi.text(
            coeff,
            ypos,
            xvars[j],
            color=colors[j],
            fontsize='large',
            horizontalalignment='center',
            verticalalignment=valign,
            zorder=4,
            backgroundcolor='white',
        )
        axi.plot(
            [coeff, coeff],
            [0, ypos],
            color=colors[j],
            lw=3,
            zorder=0,
        )
    # Do some styling of the axes:
    ymin, ymax = np.min(axi.get_ylim()), np.max(axi.get_ylim())
    axi.set_xlim(-1, 1)
    axi.set_ylim(ymin - 1, ymax + 1)
    for loc in ('left', 'right', 'top'):
        axi.spines[loc].set_visible(False)
    axi.get_yaxis().set_visible(False)
    axi.spines['bottom'].set_position('zero')
    axi.set_xticks([-1, -0.5, 0.0, 0.5, 1])
    axi.set_xticklabels([-1, -0.5, 0.0, 0.5, 1])
