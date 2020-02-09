# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module defining plots for PCA scores."""
import warnings
from matplotlib import pyplot as plt
from matplotlib.legend import Legend
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
import numpy as np
from adjustText import adjust_text
from psynlig.colors import generate_colors, generate_class_colors
from psynlig.common import (
    find_axis_intersection,
    jiggle_text,
    get_selector,
    get_text_settings,
)
from psynlig.scatter import create_scatter_legend
from psynlig.pca.loadings import _pca_1d_loadings_component


def pca_1d_scores(pca, scores, xvars=None, class_data=None, class_names=None,
                  select_components=None, add_loadings=False,
                  cmap_class=None, cmap_loadings=None, **kwargs):
    """Plot scores from a PCA model (1D).

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    scores : object like :class:`numpy.ndarray`
        The scores we are to plot.
    xvars : list of strings, optional
        Labels for the original variables. If not given,
        we will just give them names like "var1", "var2", etc.
    class_data : object like :class:`pandas.core.series.Series`, optional
        Class information for the points (if available).
    class_names : dict of strings
        A mapping from the class data to labels/names.
    select_componets : set of tuples of integers, optional
        This variable can be used to select the principal components
        we will create plot for. Note that the principal component
        numbering will here start from 1 (and not 0). If this is not
        given, all will be plotted.
    add_loadings : boolean, optional
        If this is True, we will add loadings to the plot.
    cmap_class : string or object like :class:`matplotlib.colors.Colormap`, optional
        A color map to use for classes.
    cmap_loadings : string or object like :class:`matplotlib.colors.Colormap`, optional
        A color map to use for loadings.
    kwargs : dict, optional
        Additional settings for the plotting.

    Returns
    -------
    figures : list of objects like :class:`matplotlib.figure.Figure`
        The figures created here.
    axes : list of objects like :class:`matplotlib.axes.Axes`
        The axes created here.

    """
    figures, axes = [], []
    components = pca.n_components_
    color_class, color_labels, idx_class = generate_class_colors(
        class_data, cmap=cmap_class
    )
    selector = get_selector(components, select_components, 1)

    colors = None  # Colors for loadings
    if add_loadings:
        # Create colors and variable names if needed for loadings:
        if xvars is None:
            xvars = ['var{}'.format(i + 1) for i in range(pca.n_features_)]
        colors = generate_colors(len(xvars), cmap=cmap_loadings)

    for idx1 in selector:
        # Create new figure:
        if not add_loadings:
            fig, axi = plt.subplots(constrained_layout=True)
            axl = None
        else:
            fig, (axi, axl) = plt.subplots(nrows=2, ncols=1,
                                           constrained_layout=True)
        figures.append(fig)
        axes.append((axi, axl))
        # Set up for axi:
        axi.set_xlabel('Principal component {}'.format(idx1 + 1))
        for loc in ('left', 'right', 'top'):
            axi.spines[loc].set_visible(False)
        axi.get_yaxis().set_visible(False)
        axi.spines['bottom'].set_position('zero')

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
            create_scatter_legend(
                axi, color_labels, class_names, show=True, **kwargs
            )
        if add_loadings:
            _pca_1d_loadings_component(
                axl,
                pca.components_[idx1, :],
                xvars,
                colors,
            )
    return figures, axes


def _add_loading_line_text(axi, xcoeff, ycoeff, label, color='black',
                           settings=None):
    """Add a loading line to a plot.

    This method is used when creating biplots.

    Parameters
    ----------
    axi : object like :class:`matplotlib.axes.Axes`
        The plot to add the loadings to.
    xcoeff : float
        The loading along the first principal component.
    ycoeff : float
        The loading along the second principal component,
    label : string
        The name of the original variable we are plotting the loadings
        for.
    color : string, optional
        The color to use for the text and symbol.
    settings : dict, optional
        Settings for adding the loadings. Possible settings are:

        * ``add_legend``: If this is True, we add a legend to the plot.

        * ``add_text``: If this is True, we will annotate the plot with
          labels for the variables.

        * ``text``: Additional settings for the text.

    Returns
    -------
    text : object like :class:`matplotlib.text.Text` or None
        This is the text added to the plot, if any.
    scat : object like :class:`matplotlib.collections.PathCollection` or Nine
        A scatter point used to generate a legend, if any.

    """
    text = None
    scat = None
    xlim, ylim = axi.get_xlim(), axi.get_ylim()
    # First plot the "real" length:
    biplot = True
    if settings.get('biplot', False):
        xend, yend = xcoeff, ycoeff
        scat_xend = xcoeff*1.01
        scat_yend = ycoeff*1.01
        axi.annotate('', xycoords='data', xy=(xcoeff, ycoeff), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='-|>', lw=2))
    else:
        # Add line and extend it:
        line, = axi.plot(
            [0, xcoeff], [0, ycoeff], color='black', alpha=0.8, zorder=2
        )
        xend, yend = find_axis_intersection(axi, xcoeff, ycoeff)
        if xend is None or yend is None:
            xend, yend = xcoeff, ycoeff
        axi.plot(
            [0, xend], [0, yend],
            ls=':', color=line.get_color(), alpha=line.get_alpha()
        )
        scat_xend = xend * 0.99
        scat_yend = yend * 0.99
    # Check if we should add text:
    if settings is not None and settings.get('add_text', False):
        text_settings, outline_settings = get_text_settings(
            settings.get('text', {})
        )
        text = axi.text(
            xend,
            yend,
            label,
            color=color,
            zorder=4,
            **text_settings,
        )
        if outline_settings:
            text.set_path_effects(
                [
                    path_effects.Stroke(**outline_settings),
                    path_effects.Normal()
                ]
            )
    if settings is not None and settings.get('add_legend', False):
        scat = axi.scatter(
            scat_xend,
            scat_yend,
            color=color,
            marker='X',
            label=label,
            s=200,
            edgecolor='black',
            linewidths=1,
            zorder=3,
        )
    if not biplot:
        axi.set_xlim(xlim)
        axi.set_ylim(ylim)
    return text, scat


def _add_2d_loading_lines(axi, coefficients1, coefficients2, xvars, cmap=None,
                          settings=None):
    """Add loading lines to a 2D scores plot.

    Parameters
    ----------
    axi : object like :class:`matplotlib.axes.Axes`
        The plot to add the loadings to.
    coefficients1 : object like :class:`numpy.ndarray`
        The coefficients for the first principal component.
    coefficients2 : object like :class:`numpy.ndarray`
        The coefficients for the second principal component.
    xvars : list of strings
        The labels for the original variables.
    cmap : string or object like :class:`matplotlib.colors.Colormap`, optional
        A color map to use for loadings.
    settings : dict, optional
        Settings for adding the loadings. Possible settings are:

        * ``adjust_text``: If this is set to True, we will attempt to
          adjust generated text so that they won't overlap.

        * ``jiggle_text``: If this is set to True, we will attempt to
          move text around to avoid overlap with other text boxes.
          This can be used as an alternativ to ``adjust_text``.

        * ``add_legend``: If this is True, we add a legend to the plot.

        * ``add_text``: If this is True, we will annotate the plot with
          labels for the variables.

        * ``text``: Additional settings for the text.

    Returns
    -------
    extra_artists : list of objects like :class:`matplotlib.artist.Artist`
        Artists added to the plot in this method.
    legend : object like :class:`matplotlib.legend.Legend` or None
        The extra legend created here, if any.

    """
    colors = generate_colors(len(xvars), cmap=cmap)
    texts, patches, labels = [], [], []
    legend = None
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
    return extra_artists, legend


def _pca_2d_add_loadings(fig, axi, pca, idx1, idx2, xvars=None,
                         cmap_loadings=None, loading_settings=None):
    """Add loadings to a 2D scatter plot.

    Parameters
    ----------
    fig : object like :class:`matplotlib.figure.Figure`
        Existing figure which we will add loadings to.
    axi : object like :class:`matplotlib.axes.Axes`
        Existing axis which can be used for adding loadings.
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    idx1 : integer
        The index to use for the first principal component.
    idx2 : integer
        The index to use for the second principal component.
    xvars : list of strings, optional
        Labels for the original variables. If not given, we will
        generate names like "var1", "var2", etc.
    cmap_loadings : string or object like :class:`matplotlib.colors.Colormap`, optional
        A color map to use for loadings.
    loading_settings : dict, optional
        Settings for adding the loadings.

    Returns
    -------
    extra_artists : list of objects like :class:`matplotlib.artist.Artist`
        Artists added to the plot in this method.
    legend : object like :class:`matplotlib.legend.Legend` or None
        The extra legend created here, if any.
    axj : object like :class:`matplotlib.axes.Axes` or None
        Extra axis created in case we are doing a biplot here.

    """
    extra_artists, legend = None, None
    axj = None
    if xvars is None:
        xvars = ['var{}'.format(i + 1) for i in range(pca.n_features_)]
    if loading_settings.get('biplot', False):
        # biplot mode:
        axj = fig.add_subplot(111, label='extra', frame_on=False)
        axj.xaxis.tick_top()
        axj.yaxis.tick_right()
        extra_artists, legend = _add_2d_loading_lines(
            axj,
            pca.components_[idx1, :],
            pca.components_[idx2, :],
            xvars,
            cmap=cmap_loadings,
            settings=loading_settings,
        )
        axj.set_xlim(-1, 1)
        axj.set_ylim(-1, 1)
        axj.set_xticks([-1, -0.5, 0.0, 0.5, 1])
        axj.set_yticks([-1, -0.5, 0.0, 0.5, 1])
        axj.set_xlabel('Loading for PC{}'.format(idx1 + 1))
        axj.set_ylabel('Loading for PC{}'.format(idx2 + 1))
        axj.yaxis.set_label_position('right')
        axj.xaxis.set_label_position('top')

        maxy = np.abs(axi.get_ylim()).max()
        maxx = np.abs(axi.get_xlim()).max()
        axi.set_ylim(-maxy, maxy)
        axi.set_xlim(-maxx, maxx)
    else:
        extra_artists, legend = _add_2d_loading_lines(
            axi,
            pca.components_[idx1, :],
            pca.components_[idx2, :],
            xvars,
            cmap=cmap_loadings,
            settings=loading_settings,
        )
    return extra_artists, legend, axj


def _adjust_figure_for_legend_outside(fig, axi, legend):
    """Adjust plot side in case legend is placed outside the axis.

    Parameters
    ----------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure we are adjusting.
    axi : object like :class:`matplotlib.axes.Axes`
        The axis we are adjusting.
    legend : object like :class:`matplotlib.legend.Legend`
        The legend we are adjusting for.

    """
    renderer = fig.canvas.get_renderer()
    transform_ax = axi.transAxes.inverted()
    box = legend.get_window_extent(renderer=renderer)
    box_ax = box.transformed(transform_ax)
    if box_ax.x1 - 1 > 0:
        width = box_ax.x1 - box_ax.x0
        fig.subplots_adjust(right=1.0-width)
    if box_ax.y0 - 0 < 0:
        warnings.warn(
            (
                'Legend is too hight to fit in plot, consider '
                'saving the plot with the "savefig" setting'
            ),
            RuntimeWarning
        )


def pca_2d_scores(pca, scores, xvars=None, class_data=None, class_names=None,
                  select_components=None, loading_settings=None,
                  savefig=None, cmap_class=None, cmap_loadings=None,
                  **kwargs):
    """Plot scores from a PCA model anlong two PC's.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    scores : object like :class:`numpy.ndarray`
        The scores we are to plot.
    xvars : list of strings, optional
        Labels for the original variables. If not given, we will
        generate names like "var1", "var2", etc.
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
    savefig : string, optional
        If this is given, we will here save the figure to a file.
        This is included here due to potential problems with large
        legends and displaying them in a interactive plot.
    cmap_class : string or object like :class:`matplotlib.colors.Colormap`, optional
        A color map to use for classes.
    cmap_loadings : string or object like :class:`matplotlib.colors.Colormap`, optional
        A color map to use for loadings.
    kwargs : dict, optional
        Additional settings for the plotting.

    Returns
    -------
    figures : list of objects like :class:`matplotlib.figure.Figure`
        The figures created here.
    axes : list of objects like :class:`matplotlib.axes.Axes`
        The axes created here.

    """
    figures, axes = [], []
    components = pca.n_components_
    extra_artists, legend, axj = None, None, None

    if components < 2:
        raise ValueError('Too few (< 2) principal components for a 2D plot!')

    color_class, color_labels, idx_class = generate_class_colors(
        class_data, cmap=cmap_class
    )

    selector = get_selector(components, select_components, 2)

    for idx1, idx2 in selector:
        fig, axi = plt.subplots()
        figures.append(fig)
        axes.append(axi)

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
            create_scatter_legend(
                axi, color_labels, class_names, show=True, **kwargs
            )
        axi.set_xlabel('Principal component {}'.format(idx1 + 1))
        axi.set_ylabel('Principal component {}'.format(idx2 + 1))

        if loading_settings is not None:
            # Add lines for loadings:
            extra_artists, legend, axj = _pca_2d_add_loadings(
                fig,
                axi,
                pca,
                idx1,
                idx2,
                xvars=xvars,
                cmap_loadings=cmap_loadings,
                loading_settings=loading_settings
            )
            if axj is not None:
                # Add the new axes created for loadings:
                axes[-1] = (axi, axj)
            fig.tight_layout()
            if legend is not None and savefig is None:
                _adjust_figure_for_legend_outside(fig, axi, legend)
        else:
            fig.tight_layout()
        if savefig is not None:
            fig.savefig(
                '{}_{}_{}'.format(idx1, idx2, savefig),
                bbox_extra_artists=extra_artists,
                bbox_inches='tight',
            )
    return figures, axes


def pca_3d_scores(pca, scores, class_data=None, class_names=None,
                  select_components=None, cmap_class=None, **kwargs):
    """Plot scores from a PCA model anlong two PC's.

    Parameters
    ----------
    pca : object like :class:`sklearn.decomposition._pca.PCA`
        The results from a PCA analysis.
    scores : object like :class:`numpy.ndarray`
        The scores we are to plot.
    class_data : object like :class:`pandas.core.series.Series`, optional
        Class information for the points (if available).
    class_names : dict of strings
        A mapping from the class data to labels/names.
    select_componets : set of tuples of integers, optional
        This variable can be used to select the principal components
        we will create plot for. Note that the principal component
        numbering will here start from 1 (and not 0). If this is not
        given, all will be plotted.
    cmap_class : string or object like :class:`matplotlib.colors.Colormap`, optional
        A color map to use for classes.
    kwargs : dict, optional
        Additional settings for the plotting.

    Returns
    -------
    figures : list of objects like :class:`matplotlib.figure.Figure`
        The figures created here.
    axes : list of objects like :class:`matplotlib.axes.Axes`
        The axes created here.

    """
    figures, axes = [], []
    components = pca.n_components_
    if components < 3:
        raise ValueError('Too few (< 3) principal components for a 3D plot!')
    color_class, color_labels, idx_class = generate_class_colors(
        class_data, cmap=cmap_class
    )
    selector = get_selector(components, select_components, 3)
    for idx1, idx2, idx3 in selector:
        fig = plt.figure(constrained_layout=True)
        axi = fig.add_subplot(111, projection='3d')
        figures.append(fig)
        axes.append(axi)
        if class_data is None:
            axi.scatter(
                scores[:, idx1],
                scores[:, idx2],
                scores[:, idx3],
                **kwargs
            )
        else:
            for classid, idx in idx_class.items():
                axi.scatter(
                    scores[idx, idx1],
                    scores[idx, idx2],
                    scores[idx, idx3],
                    color=color_class[classid],
                    **kwargs
                )
            create_scatter_legend(
                axi, color_labels, class_names, show=True, **kwargs
            )
        axi.set_xlabel('Principal component {}'.format(idx1 + 1), labelpad=15)
        axi.set_ylabel('Principal component {}'.format(idx2 + 1), labelpad=15)
        axi.set_zlabel('Principal component {}'.format(idx3 + 1), labelpad=15)
    return figures, axes
