# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module defining plots for PCA scores."""
import warnings
from matplotlib import pyplot as plt
from matplotlib.legend import Legend
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
import numpy as np
from adjustText import adjust_text
from psynlig.colors import generate_colors, generate_class_colors
from psynlig.common import (
    find_axis_intersection,
    jiggle_text,
    get_selector,
)
from psynlig.scatter import create_scatter_legend
from psynlig.pca.loadings import _pca_1d_loadings_component


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

    Returns
    -------
    text : object like :class:`matplotlib.text.Text` or None
        This is the text added to the plot, if any.
    scat : object like :class:`matplotlib.collections.PathCollection` or Nine
        A scatter point used to generate a legend, if any.

    """
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
    coefficients2 : object like :class:`numpy.ndarray`
        The coefficients for the second principal component.
    xvars : list of strings
        The labels for the original variables.
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

    Returns
    -------
    extra_artists : list of objects like :class:`matplotlib.artist.Artist`
        Artists added to the plot in this method.
    legend : object like :class:`matplotlib.legend.Legend` or None
        The extra legend created here, if any.

    """
    colors = generate_colors(len(xvars))
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


def pca_2d_scores(pca, scores, xvars, class_data=None, class_names=None,
                  select_components=None, loading_settings=None,
                  savefig=None, **kwargs):
    """Plot scores from a PCA model anlong two PC's.

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
    savefig : string, optional
        If this is given, we will here save the figure to a file.
        This is included here due to potential problems with large
        legends and displaying them in a interactive plot.
    kwargs : dict, optional
        Additional settings for the plotting.

    """
    components = pca.n_components_
    if components < 2:
        raise ValueError('Too few (< 2) principal components for a 2D plot!')
    color_class, color_labels, idx_class = generate_class_colors(class_data)
    selector = get_selector(components, select_components, 2)
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
            create_scatter_legend(
                axi, color_labels, class_names, show=True, **kwargs
            )
        axi.set_xlabel('Principal component {}'.format(idx1 + 1))
        axi.set_ylabel('Principal component {}'.format(idx2 + 1))
        if loading_settings is not None:
            # Add lines for loadings:
            extra_artists, legend = _add_2d_loading_lines(
                axi,
                pca.components_[idx1, :],
                pca.components_[idx2, :],
                xvars,
                settings=loading_settings,
            )
            fig.tight_layout()
            if legend is not None and savefig is None:
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
                            'saving the plot with the xxx setting'
                        ),
                        RuntimeWarning
                    )
        else:
            fig.tight_layout()
        if savefig is not None:
            fig.savefig(
                savefig,
                bbox_extra_artists=extra_artists,
                bbox_inches='tight',
            )


def pca_1d_scores(pca, scores, xvars, class_data=None, class_names=None,
                  select_components=None, add_loadings=False, **kwargs):
    """Plot scores from a PCA model (1D).

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
    add_loadings : boolean, optional
        If this is True, we will add loadings to the plot.
    kwargs : dict, optional
        Additional settings for the plotting.

    """
    components = pca.n_components_
    color_class, color_labels, idx_class = generate_class_colors(class_data)
    selector = get_selector(components, select_components, 1)
    colors = None
    if xvars is not None:
        colors = generate_colors(len(xvars))
    for idx1 in selector:
        if not add_loadings:
            fig, axi = plt.subplots()
            axl = None
        else:
            fig, (axi, axl) = plt.subplots(nrows=2, ncols=1)
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
        axi.set_xlabel('Principal component {}'.format(idx1 + 1))
        for loc in ('left', 'right', 'top'):
            axi.spines[loc].set_visible(False)
        axi.get_yaxis().set_visible(False)
        axi.spines['bottom'].set_position('zero')
        if add_loadings:
            _pca_1d_loadings_component(
                axl,
                pca.components_[idx1, :],
                xvars,
                colors,
            )
            maxx = max(np.abs(axi.get_xlim()))
            axi.set_xlim(-maxx, maxx)
        fig.tight_layout()
