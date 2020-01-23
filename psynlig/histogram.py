# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module for generating histograms plots of variables."""
import copy
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
import numpy as np
from .colors import generate_class_colors
from .common import create_fig_and_axes
from .scatter import plot_scatter


def histograms(data, variables, class_data=None, class_names=None,
               max_plots=6, ncol=3, sharex=False, sharey=False, **kwargs):
    """Generate histogram(s) from the given data.

    Parameters
    ----------
    data : object like :py:class:`pandas.core.frame.DataFrame`
        The data we are plotting.
    variables : list of strings
        The variables we are going to plot for.
    class_data : object like, optional
        Class information for the points (if available).
    class_names : dict of strings
        A mapping from the class data to labels/names.
    max_plots : integer
        Maximum number of plots to keep in one figure.
    ncol : integer
        Number of columns to use when plotting several histograms
        in the same figure.
    sharex : boolean, optional
        If True, the histograms will share the x-axis.
    sharey : boolean, optional
        If True, the histograms will share the y-axis.
    kwargs : dict, optional
        Additional settings for the plotting.

    Returns
    -------
    figures : list of objects like :py:class:`matplotlib.figure.Figure`
        The figures containing the plots.
    axes : list of objects like :py:class:`matplotlib.axes.Axes`
        The axes containing the plots.

    """
    nplots = len(variables)
    figures, axes = create_fig_and_axes(
        nplots,
        max_plots,
        ncol=ncol,
        sharex=sharex,
        sharey=sharey
    )
    fig = None
    for i, xvar in enumerate(variables):
        histogram1d(
            axes[i],
            data,
            xvar,
            class_data=class_data,
            class_names=class_names,
            **kwargs
        )
        if axes[i].figure != fig:
            fig = axes[i].figure
            axes[i].legend()
    for fig in figures:
        fig.tight_layout()
        fig.tight_layout()
    return figures, axes


def histogram1d(axi, data, variable, class_data=None, class_names=None,
                **kwargs):
    """Add a single histogram to the given axis.

    Parameters
    ----------
    axi : object like :py:class:`matplotlib.axes.Axes`
        The axis we will add the histogram to.
    data : object like :py:class:`pandas.core.frame.DataFrame`
        The data we are plotting.
    variable : list of strings
        The variable we are going to plot for.
    class_data : object like, optional
        Class information for the points (if available).
    class_names : dict of strings
        A mapping from the class data to labels/names.
    kwargs : dict, optional
        Additional settings for the plotting.

    """
    color_class, _, idx_class = generate_class_colors(class_data)
    axi.set(xlabel=variable, ylabel='Frequency')
    if class_data is None:
        axi.hist(data[variable], **kwargs)
    else:
        for class_id, idx in idx_class.items():
            axi.hist(
                data[variable][idx],
                color=color_class[class_id],
                label=class_names[class_id],
                density=False,
                **kwargs
            )


def _histogram2d_style(axi, *spines, xlim=None, ylim=None):
    """Style the given axis for 1D histograms.

    Parameters
    ----------
    axi : object like :py:class:`matplotlib.axes.Axes`
        The axis we will style.
    spines : list of strings
        The spines we will hide.
    xlim : tuple of floats, optional
        The limits for the x-axis.
    ylim : tuple of floats, optional
        The limits for the y-axis.

    """
    axi.yaxis.set_ticklabels([])
    axi.xaxis.set_ticklabels([])
    for spine in spines:
        axi.spines[spine].set_visible(False)
    axi.set(xlabel=None, ylabel=None)
    if xlim is not None:
        axi.set_xlim(xlim)
    if ylim is not None:
        axi.set_ylim(ylim)


def _histogram2d_contour(axi, data, xvar, yvar, show_contour,
                         counts, xedges, yedges, **kwargs):
    """Add a contour plot to the given axis.

    Parameters
    ----------
    axi : object like :py:class:`matplotlib.axes.Axes`
        The axis we will add the plot to.
    data : object like :py:class:`pandas.core.frame.DataFrame`
        The data we are plotting, here it is assume that we
        can select the values given by ``xvar`` and ``yvar``.
    xvar : string
        The variable to use as the x-variable.
    yvar : string
        The variable to use as the y-variable.
    show_contour : boolean or string or None
        The type of contour plot we will make. A value
        of ``filled`` will give a filled contour, anything else
        gives just the lines.
    counts : object like :py:class:`numpy.ndarray` or None
        The data points we will use to create the contour plot from.
    xedges : object like :py:class:`numpy.ndarray` or None
        X-values for the binning used to obtain ``counts``
    yedges : object like :py:class:`numpy.ndarray` or None
        Y-values for the binning used to obtain ``counts``
    kwargs : dict, optional
        Additional settings for the contour plot.

    """
    if counts is None:
        counts, xedges, yedges = np.histogram2d(
            data[xvar],
            data[yvar],
            bins=kwargs.get('histogram2d', {}).get(
                'bins', rcParams['hist.bins']
            ),
        )
    xmid = 0.5 * (xedges[1:] + xedges[:-1])
    ymid = 0.5 * (yedges[1:] + yedges[:-1])
    xmat, ymat = np.meshgrid(xmid, ymid)
    if show_contour in ('filled',):
        axi.contourf(xmat, ymat, counts.T, **kwargs.get('contour', {}))
    else:
        axi.contour(xmat, ymat, counts.T, **kwargs.get('contour', {}))
    axi.set_xlim(xmid.min(), xmid.max())
    axi.set_ylim(ymid.min(), ymid.max())


def histogram2d(data, xvar, yvar, class_data=None, class_names=None,
                show_hist=True, show_scatter=False, show_contour=False,
                **kwargs):
    """Generate a 2D histogram.

    Parameters
    ----------
    data : object like :py:class:`pandas.core.frame.DataFrame`
        The data we are plotting, here it is assume that we
        can select the values given by ``xvar`` and ``yvar``.
    xdata : string
        The variable to use as the x-variable.
    ydata : string
        The variable to use as the y-variable.
    class_data : object like, optional
        Class information for the points (if available).
    class_names : dict of strings
        A mapping from the class data to labels/names.
    show_hist : boolean, optional
        If True, we will display the 2D histogram.
    show_scatter : boolean, optional
        If True, we will show the raw data used to obtain the
        histogram.
    show_contour : boolean or string, optional
        If different from False, we will add a contour plot.
        If ``show_contour = filled`` we will then plot filled
        contours, otherwise, we will plot just the contour lines.
    kwargs : dict of dicts, optional
        Additional settings for the plot elements. It may contain the
        following keys:

        * ``histogram2d`` with settings for the 2D histogram plot.
        * ``histogram1d`` with settings for the 1D histogram plots.
        * ``scatter`` with settings for the plotting of the raw data.
        * ``contour`` with settings for the contour plot.

    """
    fig = plt.figure()
    grid = GridSpec(4, 4)
    ax_l = fig.add_subplot(grid[0, -1])
    ax_l.set_axis_off()
    ax_x = fig.add_subplot(grid[0, :-1])
    ax_y = fig.add_subplot(grid[1:, -1])
    axi = fig.add_subplot(grid[1:, :-1])
    counts = None
    xedges, yedges = [], []
    if show_hist:
        counts, xedges, yedges, _ = axi.hist2d(
            data[xvar], data[yvar], **kwargs.get('histogram2d', {})
        )
    if show_scatter:
        plot_scatter(
            data, xvar, yvar, axi=axi, class_data=class_data,
            class_names=class_names, show_legend=False,
            xy_line=False,
            trendline=False,
            **kwargs.get('scatter', {})
        )
    if show_contour:
        _histogram2d_contour(
            axi,
            data,
            xvar,
            yvar,
            show_contour,
            counts,
            xedges,
            yedges,
            **kwargs
        )
    axi.set(xlabel=xvar, ylabel=yvar)
    # Add x-histogram:
    histogram1d(
        ax_x,
        data,
        xvar,
        class_data=class_data,
        class_names=class_names,
        **kwargs.get('histogram1d', {}),
    )
    _histogram2d_style(ax_x, 'top', 'right', xlim=axi.get_xlim())
    # Add y-histogram:
    kwargs_y = copy.deepcopy(kwargs.get('histogram1d', {}))
    kwargs_y['orientation'] = 'horizontal'
    histogram1d(
        ax_y,
        data,
        yvar,
        class_data=class_data,
        class_names=class_names,
        **kwargs_y
    )
    _histogram2d_style(ax_y, 'bottom', 'right', ylim=axi.get_ylim())
    ax_y.xaxis.tick_top()
    if class_data is not None:
        handles, labels = ax_x.get_legend_handles_labels()
        ax_l.legend(handles, labels)
    fig.tight_layout()
