# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module defining plots for contributions to principal components."""
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
import numpy as np
from adjustText import adjust_text
from psynlig.colors import generate_colors
from psynlig.heatmap import plot_annotated_heatmap
from psynlig.common import (
    MARKERS,
    set_origin_axes,
    get_selector,
    get_text_settings,
)


def pca_1d_loadings(pca, xvars, select_components=None,
                    plot_type='line', cmap=None, text_settings=None):
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
    cmap : string or object like :class:`matplotlib.colors.Colormap`, optional
        A colormap to use for the components/variables.
    text_settings : dict, optional
        Additional settings for creating the text.

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
    colors = generate_colors(len(xvars), cmap=cmap)
    selector = get_selector(components, select_components, 1)
    for i in selector:
        fig, axi = plt.subplots(constrained_layout=True)
        axi.set_title('Loading coefficients for PC{}'.format(i + 1))
        coefficients = np.transpose(pca.components_[i, :])
        try:
            if plot_type.lower().startswith('bar'):
                pca_loadings_bar(axi, coefficients, xvars,
                                 plot_type=plot_type.lower())
            else:
                _pca_1d_loadings_component(axi, coefficients, xvars, colors,
                                           text_settings=text_settings)
        except AttributeError:
            _pca_1d_loadings_component(axi, coefficients, xvars, colors,
                                       text_settings=text_settings)
        figures.append(fig)
        axes.append(axi)
    return figures, axes


def _pca_1d_loadings_component(axi, coefficients, xvars, colors,
                               text_settings=None):
    """Plot the loadings for a single component in a 1D plot.

    This plot will show the components on a single line.

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
    text_settings : dict, optional
        Additional settings for creating the text.

    """
    pos_b, pos_t = 0, 0
    for i, coeff in enumerate(coefficients):
        # Add marker:
        axi.scatter(
            coeff,
            0,
            s=200,
            label=xvars[i],
            marker=MARKERS[i % len(MARKERS)],
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
        txt_settings, outline_settings = get_text_settings(
            text_settings,
            default={
                'fontsize': 'large',
                'horizontalalignment': 'center',
                'backgroundcolor': 'white',
            },
        )
        text = axi.text(
            coeff,
            ypos,
            xvars[i],
            color=colors[i],
            verticalalignment=valign,
            zorder=4,
            **txt_settings,
        )
        if outline_settings:
            text.set_path_effects(
                [
                    path_effects.Stroke(**outline_settings),
                    path_effects.Normal()
                ]
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
    axi.set_xlabel('Variables')


def pca_loadings_map(pca, xvars, val_fmt='{x:.2f}', bubble=False,
                     annotate=True, textcolors=None, plot_style=None,
                     **kwargs):
    """Show contributions from variables to PC's in a heat map.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    xvars : list of strings
        The labels for the original variables.
    val_fmt : string, optional
        The format of the annotations inside the heat map.
    bubble : boolean, optional
        If True, we will draw bubbles to indicate the size of the
        given data points.
    annotate : boolean, optional
        If True, we will write annotate the plot with values for the
        contributions.
    textcolors : list of strings, optional
        Colors used for the text. The number of colors provided defines
        a binning for the data values, and values are colored with the
        corresponding color. If no colors are provided, all are colored
        black.
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
        val_fmt=val_fmt,
        annotate=annotate,
        bubble=bubble,
        textcolors=textcolors,
        **kwargs
    )
    return fig1, ax1


def pca_2d_loadings(pca, xvars, select_components=None, adjust_labels=False,
                    cmap=None, style='box', text_settings=None):
    """Show loadings for two principal compoents.

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
    cmap : string or object like :class:`matplotlib.colors.Colormap`, optional
        A colormap to use for the components/variables.
    style : string, optional
        This option changes the styling of the plot.
        For style ``box``, we show the axes as a normal matplotlib
        figure with inserted lines showing x=0 and y=0.
        For the style 'center' we place the x-axis and y-axis at
        the origin.
    text_settings : dict, optional
        Additional settings for creating the text.

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
    colors = generate_colors(len(xvars), cmap=cmap)
    selector = get_selector(components, select_components, 2)
    for idx1, idx2 in selector:
        fig, axi = plt.subplots(constrained_layout=True)
        coefficients1 = np.transpose(pca.components_[idx1, :])
        coefficients2 = np.transpose(pca.components_[idx2, :])
        _pca_2d_loadings_component(
            axi,
            coefficients1,
            coefficients2,
            xvars,
            colors,
            adjust_labels=adjust_labels,
            text_settings=text_settings
        )
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
        figures.append(fig)
        axes.append(axi)
    return figures, axes


def _pca_2d_loadings_component(axi, coefficients1, coefficients2,
                               xvars, colors, adjust_labels=False,
                               text_settings=None):
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
    text_settings : dict, optional
        Additional settings for creating the text.

    """
    texts, points = [], []
    axi.set_aspect('equal')
    for i, (coeff1, coeff2) in enumerate(zip(coefficients1, coefficients2)):
        scat = axi.scatter(
            coeff1,
            coeff2,
            s=200,
            label=xvars[i],
            marker=MARKERS[i % len(MARKERS)],
            color=colors[i],
        )
        points.append(scat)
        txt_settings, outline_settings = get_text_settings(
            text_settings,
            default={
                'fontsize': 'large',
            },
        )
        if txt_settings.get('show', True):
            text = axi.text(
                coeff1,
                coeff2,
                xvars[i],
                color=colors[i],
                **txt_settings,
            )
            if outline_settings:
                text.set_path_effects(
                    [
                        path_effects.Stroke(**outline_settings),
                        path_effects.Normal()
                    ]
                )
            texts.append(text)
    if adjust_labels and texts:
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


def pca_3d_loadings(pca, xvars, select_components=None, cmap=None,
                    text_settings=None):
    """Show contributions to three principal components.

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
    cmap : string or object like :class:`matplotlib.colors.Colormap`, optional
        A colormap to use for the components/variables.
    text_settings : dict, optional
        Additional settings for creating the text.

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
    colors = generate_colors(len(xvars), cmap=cmap)
    selector = get_selector(components, select_components, 3)
    for idx1, idx2, idx3 in selector:
        fig = plt.figure(constrained_layout=True)
        axi = fig.add_subplot(111, projection='3d')
        axi.set_xlabel('Principal component {}'.format(idx1 + 1), labelpad=15)
        axi.set_ylabel('Principal component {}'.format(idx2 + 1), labelpad=15)
        axi.set_zlabel('Principal component {}'.format(idx3 + 1), labelpad=15)
        coefficients1 = np.transpose(pca.components_[idx1, :])
        coefficients2 = np.transpose(pca.components_[idx2, :])
        coefficients3 = np.transpose(pca.components_[idx3, :])
        _pca_3d_loadings_component(axi, coefficients1, coefficients2,
                                   coefficients3, xvars, colors,
                                   text_settings=text_settings)
        figures.append(fig)
        axes.append(axi)
    return figures, axes


def _pca_3d_loadings_component(axi, coefficients1, coefficients2,
                               coefficients3, xvars, colors,
                               text_settings=None):
    """Show contributions to three principal components.

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
    text_settings : dict, optional
        Additional settings for creating the text.

    """
    coeffs = zip(coefficients1, coefficients2, coefficients3)
    for i, (coeff1, coeff2, coeff3) in enumerate(coeffs):
        axi.scatter(
            coeff1,
            coeff2,
            coeff3,
            s=200,
            label=xvars[i],
            marker=MARKERS[i % len(MARKERS)],
            color=colors[i],
        )
        txt_settings, outline_settings = get_text_settings(
            text_settings,
            default={'fontsize': 'xx-large'},
        )
        if txt_settings.get('show', True):
            text = axi.text(
                coeff1 + 0.02,
                coeff2 + 0.02,
                coeff3 + 0.02,
                xvars[i],
                color=colors[i],
                **txt_settings,
            )
            if outline_settings:
                text.set_path_effects(
                    [
                        path_effects.Stroke(**outline_settings),
                        path_effects.Normal()
                    ]
                )
    axi.set_xlim(-1, 1)
    axi.set_ylim(-1, 1)
    axi.set_zlim(-1, 1)
    axi.plot([-1, 1], [0, 0], ls=':', color='#262626', alpha=0.8, lw=3)
    axi.plot([0, 0], [-1, 1], ls=':', color='#262626', alpha=0.8)
    axi.plot([0, 0], [0, 0], [-1, 1], ls=':', color='#262626', alpha=0.8)
