# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module defining plots for PCA results."""
from itertools import combinations
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.legend import Legend
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
import numpy as np
from adjustText import adjust_text
from .colors import generate_colors, generate_class_colors
from .heatmap import plot_annotated_heatmap
from .common import (
    MARKERS,
    set_origin_axes,
    find_axis_intersection,
    jiggle_text,
)
from .scatter import create_scatter_legend


def pca_explained_variance(pca, axi=None, **kwargs):
    """Plot the explained variance as function of PCA components.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    axi : object like :class:`matplotlib.axes.Axes`, optional
        If given, the plot will be added to the specified axis.
        Otherwise, a new axis (and figure) will be created here.
    kwargs : dict
        Additional settings for plotting explained variance.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure containing the plot, if the figure is created
        here. Oterwise, it is None.
    axi : object like :class:`matplotlib.axes.Axes`
        The axis containing the plot.

    """
    fig = None
    if axi is None:
        fig, axi = plt.subplots(nrows=1, ncols=1)
    var = [0] + list(np.cumsum(pca.explained_variance_ratio_))
    comp = range(0, len(var))
    axi.plot(comp, var, **kwargs)
    axi.axhline(y=1, color='black', ls=':')
    axi.set(xlabel='Number of components',
            ylabel='Explained variance (fraction)')
    axi.xaxis.set_major_locator(MaxNLocator(integer=True))
    if fig is not None:
        fig.tight_layout()
    return fig, axi


def pca_residual_variance(pca, axi=None, **kwargs):
    """Plot the residual variance as function of PCA components.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    axi : object like :class:`matplotlib.axes.Axes`, optional
        If given, the plot will be added to the specified axis.
        Otherwise, a new axis (and figure) will be created here.
    kwargs : dict
        Additional settings for plotting explained variance.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure containing the plot, if the figure is created
        here. Oterwise, it is None.
    axi : object like :class:`matplotlib.axes.Axes`
        The axis containing the plot.

    """
    fig = None
    if axi is None:
        fig, axi = plt.subplots(nrows=1, ncols=1)
    var = 1 - np.array([0] + list(np.cumsum(pca.explained_variance_ratio_)))
    comp = range(0, len(var))
    axi.axhline(y=0, color='black', ls=':')
    axi.plot(comp, var, **kwargs)
    axi.set(xlabel='Number of components',
            ylabel='Residual variance (fraction)')
    axi.xaxis.set_major_locator(MaxNLocator(integer=True))
    if fig is not None:
        fig.tight_layout()
    return fig, axi


def pca_scree(pca, axi=None, **kwargs):
    """Plot the eigenvalues as function of PCA components.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    axi : object like :class:`matplotlib.axes.Axes`, optional
        If given, the plot will be added to the specified axis.
        Otherwise, a new axis (and figure) will be created here.
    kwargs : dict
        Additional settings for the plotting.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure containing the plot, if the figure is created
        here. Oterwise, it is None.
    axi : object like :class:`matplotlib.axes.Axes`
        The axis containing the plot.

    """
    fig = None
    if axi is None:
        fig, axi = plt.subplots(nrows=1, ncols=1)
    eigenvalues = pca.explained_variance_
    comp = range(1, len(eigenvalues) + 1)
    axi.plot(comp, eigenvalues, **kwargs)
    axi.set(xlabel='Principal component',
            ylabel='Eigenvalue')
    axi.xaxis.set_major_locator(MaxNLocator(integer=True))
    axi.set_xlim(min(comp) - 0.25, max(comp) + 0.25)
    if fig is not None:
        fig.tight_layout()
    return fig, axi


def pca_explained_variance_bar(pca, axi=None, **kwargs):
    """Plot the explained variance per principal component.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    axi : object like :class:`matplotlib.axes.Axes`, optional
        If given, the plot will be added to the specified axis.
        Otherwise, a new axis (and figure) will be created here.
    kwargs : dict
        Additional settings for plotting explained variance.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure containing the plot, if the figure is created
        here. Oterwise, it is None.
    axi : object like :class:`matplotlib.axes.Axes`
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
        ylabel='Explained variance (fraction) per component',
    )
    if fig is not None:
        fig.tight_layout()
    return fig, axi


def pca_explained_variance_pie(pca, axi=None, tol=1.0e-3):
    """Show the explained variance as function of PCA components.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    axi : object like :class:`matplotlib.axes.Axes`, optional
        If given, the plot will be added to the specified axis.
        Otherwise, a new axis (and figure) will be created here.
    tol : float, optional
        A tolerance for the missing variance. If the unexplained
        variance is less than this tolerance, it will not be shown.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure containing the plot, if the figure is created
        here. Oterwise, it is None.
    axi : object like :class:`matplotlib.axes.Axes`
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


def pca_1d_loadings(pca, xvars, select_components=None, plot_type='line'):
    """Plot the loadings from a PCA in a 1D plot.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    xvars : list of strings
        Labels for the original variables.
    select_componets : set of integers, optional
        This variable can be used to select the principal components
        we will create plot for. Note that the principal component
        numbering will here start from 1 (and not 0). If this is not
        given, all will be plotted.
    plot_type : string, optional
        Select the kind of plot we will be making. Possible values are:

        * ``line``: For generating a 1D line with contributions.

        * ``bar``: For generating a bar plot of the contributions.

        * ``bar-square``: For generating a bar plot of the squared
          contributions.

        * ``bar-absolute``: For generating a bar plot of the absolute
          value of contributions.

    Returns
    -------
    figures : list of objects like :class:`matplotlib.figure.Figure`
        The figures containing the plots.
    axes : list of objects like :class:`matplotlib.axes.Axes`
        The axes containing the plots.

    """
    figures = []
    axes = []
    components = pca.n_components_
    colors = generate_colors(len(xvars))
    selector = _get_selector(components, select_components, 1)
    for i in selector:
        fig, axi = plt.subplots()
        axi.set_title('Principal component {}'.format(i + 1))
        coefficients = np.transpose(pca.components_[i, :])
        try:
            if plot_type.lower().startswith('bar'):
                pca_loadings_bar(axi, coefficients, xvars,
                                 plot_type=plot_type.lower())
            else:
                pca_1d_loadings_component(axi, coefficients, xvars, colors)
        except AttributeError:
            pca_1d_loadings_component(axi, coefficients, xvars, colors)
        fig.tight_layout()
        figures.append(fig)
        axes.append(axi)
    return figures, axes


def pca_1d_loadings_component(axi, coefficients, xvars, colors):
    """Plot the loadings for a single component in a 1D plot.

    Parameters
    ----------
    axi : object like :class:`matplotlib.axes.Axes`
        The plot we will add the loadings to.
    coefficients : object like :class:`numpy.ndarray`
        The coefficients we are to show.
    xvars : list of strings
        Labels for the original variables.
    colors : list of floats or strings
        The colors used for the different labels.

    """
    pos_b, pos_t = 0, 0
    for i, coeff in enumerate(coefficients):
        # Add marker:
        axi.scatter(
            coeff,
            0,
            s=200,
            label=xvars[i],
            marker=MARKERS.get(i, 'o'),
            color=colors[i],
            zorder=4,
        )
        if i % 2 == 0:
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
            xvars[i],
            color=colors[i],
            fontsize='large',
            horizontalalignment='center',
            verticalalignment=valign,
            zorder=4,
            backgroundcolor='white',
        )
        axi.plot(
            [coeff, coeff],
            [0, ypos],
            color=colors[i],
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


def pca_loadings_bar(axi, coefficients, xvars, plot_type='bar'):
    """Plot the loadings for a single component in a bar plot.

    Parameters
    ----------
    axi : object like :class:`matplotlib.axes.Axes`
        The plot we will add the loadings to.
    coefficients : object like :class:`numpy.ndarray`
        The coefficients we are to show.
    xvars : list of strings
        Labels for the original variables.
    plot_type : string, optional
        Selects the type of plot we are making.

    """
    xpos = range(len(coefficients))
    if plot_type == 'bar-square':
        yval = coefficients**2
        ylabel = 'Squared coefficients'
    elif plot_type == 'bar-absolute':
        yval = np.abs(coefficients)
        ylabel = 'Absolute value of coefficients'
    else:
        yval = coefficients
        ylabel = 'Coefficient'
    axi.set_ylabel(ylabel)
    axi.axhline(y=0, ls=':', color='#262626')
    axi.bar(xpos, yval)
    axi.set_xticks(xpos)
    axi.set_xticklabels(
        xvars,
        rotation='vertical',
    )


def pca_loadings_map(pca, xvars, val_fmt='{x:.2f}', bubble=False,
                     annotate=True, textcolors=None, plot_style=None,
                     **kwargs):
    """Plot the contributions from variables to the principal components.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    xvars : list of strings
        The labels for the original variables.
    val_fmt : string, optional
        The format of the annotations inside the heat map.
    textcolors : list of strings, optional
        Colors used for the text. The number of colors provided defines
        a binning for the data values, and values are colored with the
        corresponding color. If no colors are provided, all are colored
        black.
    bubble : boolean, optional
        If True, we will draw bubbles to indicate the size of the
        given data points.
    annotate : boolean, optional
        If True, we will write the values as text in the plot.
    plot_style : string, optional
        Determines how the cofficients are plotted:

        * ``absolute``: The absolute value of the coefficients will
          be plotted.

        * ``squared``: The squared value of the coefficients will be
          plotted.

        Otherwise, the actual value of the coefficients will be used.

    **kwargs : dict, optional
        Arguments used for drawing the heat map.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure in which the heatmap is plotted.
    axi : object like :class:`matplotlib.axes.Axes`
        The axis to which the heatmap is added.

    """
    components = pca.components_
    label = 'Coefficients'
    # rows: PC, columns: variables
    comp = ['PC{}'.format(i + 1) for i in range(pca.n_components_)]
    try:
        if plot_style.lower() == 'absolute':
            components = np.abs(components)
            label = 'Absolute coefficients'
        elif plot_style.lower() == 'squared':
            components = components**2
            label = 'Squared coefficients'
    except AttributeError:
        pass
    fig1, ax1 = plot_annotated_heatmap(
        components.T,
        xvars,
        comp,
        cbarlabel=label,
        annotate=annotate,
        bubble=bubble,
        textcolors=textcolors,
        **kwargs
    )
    return fig1, ax1


def pca_2d_loadings(pca, xvars, select_components=None, adjust_labels=False,
                    style='box'):
    """Plot the loadings from a PCA in a 2D plot.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    xvars : list of strings
        Labels for the original variables.
    select_componets : set of tuples of integers, optional
        This variable can be used to select the principal components
        we will create plot for. Note that the principal component
        numbering will here start from 1 (and not 0). If this is not
        given, all will be plotted.
    adjust_labels : boolean, optional
        If this is True, we will try to optimize the position of the
        labels so that they wont overlap.
    style : string, optional
        This option changes the styling of the plot.
        For style ``box``, we show the axes as a normal matplotlib
        figure with inserted lines showing x=0 and y=0.
        For the style 'center' we place the x-axis and y-axis at
        the origin.

    Returns
    -------
    figures : list of objects like :class:`matplotlib.figure.Figure`
        The figures containing the plots.
    axes : list of objects like :class:`matplotlib.axes.Axes`
        The axes containing the plots.

    """
    figures = []
    axes = []
    components = pca.n_components_
    if components < 2:
        raise ValueError('Too few (< 2) principal components for a 2D plot!')
    colors = generate_colors(len(xvars))
    selector = _get_selector(components, select_components, 2)
    for idx1, idx2 in selector:
        fig, axi = plt.subplots()
        coefficients1 = np.transpose(pca.components_[idx1, :])
        coefficients2 = np.transpose(pca.components_[idx2, :])
        pca_2d_loadings_component(axi, coefficients1, coefficients2,
                                  xvars, colors, adjust_labels=adjust_labels)
        axi.set(
            xlabel='Principal component {}'.format(idx1 + 1),
            ylabel='Principal component {}'.format(idx2 + 1),
        )
        if style == 'box':
            axi.axhline(y=0, ls=':', color='#262626', alpha=0.6)
            axi.axvline(x=0, ls=':', color='#262626', alpha=0.6)
        elif style == 'center':
            set_origin_axes(
                axi,
                'PC{}'.format(idx1 + 1),
                'PC{}'.format(idx2 + 1),
                fontsize='x-large',
            )
            axi.set_xticks([-1, -0.5, 0.5, 1])
            axi.set_yticks([-1, -0.5, 0.5, 1])
        else:
            # Do not do any styling.
            pass
        fig.tight_layout()
        figures.append(fig)
        axes.append(axi)
    return figures, axes


def pca_2d_loadings_component(axi, coefficients1, coefficients2,
                              xvars, colors, adjust_labels=False):
    """Plot the loadings for two components in a 2D plot.

    Parameters
    ----------
    axi : object like :class:`matplotlib.axes.Axes`
        The plot we will add the loadings to.
    coefficients1 : object like :class:`numpy.ndarray`
        The coefficients for the first principal component.
    coefficients2 : object like :class:`numpy.ndarray`
        The coefficients for the second principal component.
    xvars : list of strings
        Labels for the original variables.
    colors : list of floats or strings
        The colors used for the different labels.
    adjust_labels : boolean, optional
        If this is True, we will try to optimize the position of the
        labels so that they wont overlap.

    """
    texts, points = [], []
    axi.set_aspect('equal')
    for i, (coeff1, coeff2) in enumerate(zip(coefficients1, coefficients2)):
        scat = axi.scatter(
            coeff1,
            coeff2,
            s=200,
            label=xvars[i],
            marker=MARKERS.get(i, 'o'),
            color=colors[i],
        )
        points.append(scat)
        text = axi.text(
            coeff1,
            coeff2,
            xvars[i],
            color=colors[i],
            fontsize='large',
        )
        texts.append(text)
    if adjust_labels:
        adjust_text(
            texts,
            add_objects=points,
            expand_objects=(1.2, 1.2),
            expand_text=(1.2, 1.2),
            expand_points=(1.2, 1.2),
            force_text=(0.25, 0.25),
            force_points=(0.5, 0.5),
            force_objects=(0.25, 0.25),
        )
    axi.set_xlim(-1, 1)
    axi.set_ylim(-1, 1)


def _get_selector(components, select_components, combi):
    """Get a selector for components.

    Parameters
    ----------
    components : integer
        The number of components we are selecting from,
    select_components : iterable or None
        The items we are to pick. If this is None, we select
        all combinations.
    combi : integer
        The number of combinations of the components we
        are selecting.

    Returns
    -------
    selector : generator
        A generator which gives the indices for the components
        we are to select.

    """
    if select_components is None:
        if combi == 1:
            selector = range(components)
        else:
            selector = combinations(range(components), combi)
    else:
        if combi == 1:
            selector = (i - 1 for i in select_components)
        else:
            selector = (
                (i - 1 for i in j) for j in select_components
            )
    return selector


def pca_3d_loadings(pca, xvars, select_components=None):
    """Plot the loadings from a PCA in a 3D plot.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    xvars : list of strings
        Labels for the original variables.
    select_componets : set of tuples of integers, optional
        This variable can be used to select the principal components
        we will create plot for. Note that the principal component
        numbering will here start from 1 (and not 0). If this is not
        given, all will be plotted.

    Returns
    -------
    figures : list of objects like :class:`matplotlib.figure.Figure`
        The figures containing the plots.
    axes : list of objects like :class:`matplotlib.axes.Axes`
        The axes containing the plots.

    """
    figures = []
    axes = []
    components = pca.n_components_
    if components < 3:
        raise ValueError('Too few (< 3) principal components for a 3D plot!')
    colors = generate_colors(len(xvars))
    selector = _get_selector(components, select_components, 3)
    for idx1, idx2, idx3 in selector:
        fig = plt.figure()
        axi = fig.add_subplot(111, projection='3d')
        axi.set_xlabel('Principal component {}'.format(idx1 + 1), labelpad=15)
        axi.set_ylabel('Principal component {}'.format(idx2 + 1), labelpad=15)
        axi.set_zlabel('Principal component {}'.format(idx3 + 1), labelpad=15)
        coefficients1 = np.transpose(pca.components_[idx1, :])
        coefficients2 = np.transpose(pca.components_[idx2, :])
        coefficients3 = np.transpose(pca.components_[idx3, :])
        pca_3d_loadings_component(axi, coefficients1, coefficients2,
                                  coefficients3, xvars, colors)
        fig.tight_layout()
        figures.append(fig)
        axes.append(axi)
    return figures, axes


def pca_3d_loadings_component(axi, coefficients1, coefficients2,
                              coefficients3, xvars, colors):
    """Plot the loadings for two components in a 2D plot.

    Parameters
    ----------
    axi : object like :class:`matplotlib.axes.Axes`
        The plot we will add the loadings to.
    coefficients1 : object like :class:`numpy.ndarray`
        The coefficients for the first principal component.
    coefficients2 : object like :class:`numpy.ndarray`
        The coefficients for the second principal component.
    coefficients3 : object like :class:`numpy.ndarray`
        The coefficients for the second principal component.
    xvars : list of strings
        Labels for the original variables.
    colors : list of floats or strings
        The colors used for the different labels.

    """
    coeffs = zip(coefficients1, coefficients2, coefficients3)
    for i, (coeff1, coeff2, coeff3) in enumerate(coeffs):
        axi.scatter(
            coeff1,
            coeff2,
            coeff3,
            s=200,
            label=xvars[i],
            marker=MARKERS.get(i, 'o'),
            color=colors[i],
        )
        axi.text(
            coeff1 + 0.02,
            coeff2 + 0.02,
            coeff3 + 0.02,
            xvars[i],
            color=colors[i],
            fontsize='x-large',
        )
    axi.set_xlim(-1, 1)
    axi.set_ylim(-1, 1)
    axi.set_zlim(-1, 1)
    axi.plot([-1, 1], [0, 0], ls=':', color='#262626', alpha=0.8, lw=3)
    axi.plot([0, 0], [-1, 1], ls=':', color='#262626', alpha=0.8)
    axi.plot([0, 0], [0, 0], [-1, 1], ls=':', color='#262626', alpha=0.8)


def _add_loading_line_text(axi, xcoeff, ycoeff, label, color='black',
                           settings=None):
    """Add a loading line to a plot."""
    text = None
    scat = None
    # First plot the "real" length:
    line, = axi.plot([0, xcoeff], [0, ycoeff], color='black', alpha=0.8)
    # Then extend the line so that the length is the given length:
    xlim, ylim = axi.get_xlim(), axi.get_ylim()
    xend, yend = find_axis_intersection(axi, xcoeff, ycoeff)
    if xend is None or yend is None:
        xend, yend = xcoeff, ycoeff
    axi.plot(
        [0, xend], [0, yend],
        ls=':', color=line.get_color(), alpha=line.get_alpha()
    )
    # Check if we should add text:
    if settings is not None and settings.get('add_text', False):
        text = axi.text(
            xend,
            yend,
            label,
            weight='bold',
            horizontalalignment='left',
            verticalalignment='center',
            color=color,
            fontsize='large',
        )
    if settings is not None and settings.get('add_legend', False):
        scat = axi.scatter(
            xend * 0.99,
            yend * 0.99,
            color=color,
            marker='X',
            label=label,
            s=200
        )
    axi.set_xlim(xlim)
    axi.set_ylim(ylim)
    return text, scat


def _add_2d_loading_lines(axi, coefficients1, coefficients2, xvars,
                          settings=None):
    """Add loading lines to a 2D scores plot.

    Parameters
    ----------
    axi : object like :class:`matplotlib.axes.Axes`
        The plot to add the loadings to.
    coefficients1 : object like :class:`numpy.ndarray`
        The coefficients for the first principal component.
    coefficients1 : object like :class:`numpy.ndarray`
        The coefficients for the second principal component.
    xvars : list of strings
        The labels for the original variables.
    settings : dict, optional
        Settings for adding the loadings.

    """
    colors = generate_colors(len(xvars))
    texts, patches, labels = [], [], []
    for i, (coeff_x, coeff_y) in enumerate(zip(coefficients1, coefficients2)):
        text, scat = _add_loading_line_text(
            axi,
            coeff_x,
            coeff_y,
            xvars[i],
            color=colors[i],
            settings=settings,
        )
        if scat is not None:
            patches.append(scat)
            labels.append(xvars[i])
        if text is not None:
            texts.append(text)
    extra_artists = []
    if settings is not None:
        if texts:
            extra_artists += texts
        if settings.get('adjust_text', False) and texts:
            adjust_text(texts)
        if settings.get('jiggle_text', False) and texts:
            jiggle_text(axi, texts)
        if settings.get('add_legend', False) and patches and labels:
            legend = Legend(
                axi,
                patches,
                labels,
                title='Variables:',
                title_fontsize='large',
                bbox_to_anchor=(1, 1),
            )
            axi.add_artist(legend)
            extra_artists += [legend]
    return extra_artists


def pca_2d_scores(pca, scores, xvars, class_data=None, class_names=None,
                  select_components=None, loading_settings=None, **kwargs):
    """Plot scores from a PCA model.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    scores : object like :class:`numpy.ndarray`
        The scores we are to plot.
    xvars : list of strings
        Labels for the original variables.
    class_data : object like :class:`pandas.core.series.Series`, optional
        Class information for the points (if available).
    class_names : dict of strings
        A mapping from the class data to labels/names.
    select_componets : set of tuples of integers, optional
        This variable can be used to select the principal components
        we will create plot for. Note that the principal component
        numbering will here start from 1 (and not 0). If this is not
        given, all will be plotted.
    loading_settings : dict, optional
        Settings for adding the loadings.
    kwargs : dict, optional
        Additional settings for the plotting.

    """
    components = pca.n_components_
    if components < 2:
        raise ValueError('Too few (< 2) principal components for a 2D plot!')
    color_class, color_labels, idx_class = generate_class_colors(class_data)
    selector = _get_selector(components, select_components, 2)
    for idx1, idx2 in selector:
        fig, axi = plt.subplots()
        if class_data is None:
            axi.scatter(scores[:, idx1], scores[:, idx2], **kwargs)
        else:
            for classid, idx in idx_class.items():
                axi.scatter(
                    scores[idx, idx1],
                    scores[idx, idx2],
                    color=color_class[classid],
                    **kwargs
                )
            patches, labels = create_scatter_legend(
                axi, color_labels, class_names, show=True, **kwargs
            )
        axi.set_xlabel('Principal component {}'.format(idx1 + 1))
        axi.set_ylabel('Principal component {}'.format(idx2 + 1))
        if loading_settings is not None:
            # Add lines for loadings:
            extra_artists = _add_2d_loading_lines(
                axi,
                pca.components_[idx1, :],
                pca.components_[idx2, :],
                xvars,
                settings=loading_settings,
            )
        fig.tight_layout()


def pca_1d_scores(pca, scores, xvars, class_data=None, class_names=None,
                  select_components=None, **kwargs):
    """Plot scores from a PCA model (1D)

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    scores : object like :class:`numpy.ndarray`
        The scores we are to plot.
    xvars : list of strings
        Labels for the original variables.
    class_data : object like :class:`pandas.core.series.Series`, optional
        Class information for the points (if available).
    class_names : dict of strings
        A mapping from the class data to labels/names.
    select_componets : set of tuples of integers, optional
        This variable can be used to select the principal components
        we will create plot for. Note that the principal component
        numbering will here start from 1 (and not 0). If this is not
        given, all will be plotted.
    kwargs : dict, optional
        Additional settings for the plotting.

    """
    components = pca.n_components_
    color_class, color_labels, idx_class = generate_class_colors(class_data)
    selector = _get_selector(components, select_components, 1)
    for idx1 in selector:
        fig, axi = plt.subplots()
        if class_data is None:
            axi.scatter(
                scores[:, idx1],
                np.zeros_like(scores[:, idx1]),
                **kwargs
            )
        else:
            for classid, idx in idx_class.items():
                axi.scatter(
                    scores[idx, idx1],
                    np.zeros_like(scores[idx, idx1]),
                    color=color_class[classid],
                    **kwargs
                )
            patches, labels = create_scatter_legend(
                axi, color_labels, class_names, show=True, **kwargs
            )
        axi.set_xlabel('Principal component {}'.format(idx1 + 1))
        for loc in ('left', 'right', 'top'):
            axi.spines[loc].set_visible(False)
        axi.get_yaxis().set_visible(False)
        axi.spines['bottom'].set_position('zero')
        fig.tight_layout()
