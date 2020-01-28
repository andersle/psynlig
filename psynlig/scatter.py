# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module for generating scatter plots of variables."""
from itertools import combinations
import pprint
import warnings
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
from scipy.special import comb
from .colors import generate_class_colors
from .common import (
    create_fig_and_axes,
    add_xy_line,
    add_trendline,
    iqr_outlier,
)


_WARNING_MAX_PLOTS = (
    'This will generate {0} plots. If you want to generate '
    'all these plots, rerun the function with the '
    'argument "max_plots={0}".'
)


def create_scatter_legend(axi, color_labels, class_names, show=False,
                          **kwargs):
    """Generate a legend for a scatter plot with class labels.

    Parameters
    ----------
    axi : object like :class:`matplotlib.axes.Axes`
        The axes we will add the legend for.
    color_labels : dict of objects like :class:`numpy.ndarray`
        Colors for the different classes.
    color_names : dict of strings
        Names for the classes.
    show : boolean, optional
        If True, we will add the legend here.
    kwargs : dict, optional
        Additional arguments passed to the scatter method. Used
        here to get a consistent styling.

    Returns
    -------
    patches : list of objects like :class:`matplotlib.artist.Artist`
        The items we will create a legend for.
    labels : list of strings
        The labels for the legend.

    """
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
    if show:
        axi.legend(patches, labels, ncol=1)
    return patches, labels


def plot_scatter(data, xvar, yvar, axi=None,
                 class_data=None, class_names=None,
                 highlight=None, **kwargs):
    """Make a 2D scatter plot of the given data.

    Parameters
    ----------
    data : object like :class:`pandas.core.frame.DataFrame`
        The data we are plotting.
    xvar : string
        The column to use as the x-variable.
    yvar : string
        The column to use as the y-variable.
    axi : object like :class:`matplotlib.axes.Axes`, optional
        An axis to add the plot to. If this is not provided,
        a new axis (and figure) will be created here.
    class_data : object like :class:`pandas.core.series.Series`, optional
        Class information for the points (if available).
    class_names : dict of strings
        A mapping from the class data to labels/names.
    highlight : list of integers, optional
        This can be used to highlight certain points in the plot.
    kwargs : dict, optional
        Additional settings for the plotting.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure containing the plot.
    axi : object like :class:`matplotlib.axes.Axes`
        The axis containing the plot.
    patches : list of objects like :class:`matplotlib.artist.Artist`
        The items we will create a legend for.
    labels : list of strings
        The labels for the legend.

    """
    patches, labels = [], []
    color_class, color_labels, idx_class = generate_class_colors(class_data)
    fig = None
    if axi is None:
        fig, axi = plt.subplots()
    if xvar is None:
        axi.set(xlabel='Data point no.', ylabel=yvar)
        xdata = np.arange(len(data[yvar]))
    else:
        axi.set(xlabel=xvar, ylabel=yvar)
        xdata = data[xvar]
    ydata = data[yvar]

    if class_data is None:
        axi.scatter(xdata, ydata, **kwargs)
    else:
        for class_id, idx in idx_class.items():
            axi.scatter(
                xdata[idx],
                ydata[idx],
                color=color_class[class_id],
                **kwargs
            )
        patches, labels = create_scatter_legend(
            axi, color_labels, class_names, **kwargs
        )
    if highlight is not None:
        axi.scatter(xdata[highlight], ydata[highlight], marker='X')
    if fig is not None:
        fig.tight_layout()
    return fig, axi, patches, labels


def generate_1d_scatter(data, variables, class_data=None, class_names=None,
                        max_plots=6, ncol=3, sharex=False, sharey=False,
                        show_legend=True, outliers=False,
                        **kwargs):
    """Generate 1D scatter plots from the given data and variables.

    Parameters
    ----------
    data : object like :class:`pandas.core.frame.DataFrame`
        The data we will plot here.
    variables : list of strings
        The variables we will generate scatter plots for.
    class_data : object like :class:`pandas.core.series.Series`, optional
        Class information for the points (if available).
    class_names : dict of strings, optional
        A mapping from the class data to labels/names.
    max_plots : integer, optional
        The maximum number of plots in a figure.
    ncol : integer, optional
        The number of columns to use in a figure.
    sharex : boolean, optional
        If True, the scatter plots will share the x-axis.
    sharey : boolean, optional
        If True, the scatter plots will share the y-axis.
    show_legend : boolean, optional
        If True, we will create a legend here and show it.
    outliers : boolean, optional
        If True, we will try to mark outliers in the plot.
    kwargs : dict, optional
        Additional arguments used for the plotting.

    Returns
    -------
    figures : list of objects like :class:`matplotlib.figure.Figure`
        The figures containing the plots.
    axes : list of objects like :class:`matplotlib.axes.Axes`
        The axes containing the plots.

    """
    nplots = len(variables)
    figures, axes = create_fig_and_axes(
        nplots, max_plots, ncol=ncol, sharex=sharex, sharey=sharey
    )

    outlier_points = {}
    bounds = [{}, {}]
    if outliers:
        _, outlier_points, bounds = iqr_outlier(data, variables)

    fig = None
    for i, yvar in enumerate(variables):
        show_legend = False
        if axes[i].figure != fig:
            fig = axes[i].figure
            show_legend = True
        highlight = None

        if outliers:
            highlight = outlier_points.get(yvar, None)
            if highlight is not None:
                print('Possible outliers for "{}":'.format(yvar))
                pprint.pprint(highlight)

        _, _, patches, labels = plot_scatter(
            data,
            None,
            yvar,
            axi=axes[i],
            class_data=class_data,
            class_names=class_names,
            highlight=highlight,
            **kwargs
        )
        if outliers:
            lower = bounds[0].get(yvar, None)
            upper = bounds[1].get(yvar, None)
            if lower is not None:
                axes[i].axhline(y=lower, ls=':', color='#262626')
            if upper is not None:
                axes[i].axhline(y=upper, ls=':', color='#262626')

        if show_legend and patches and labels:
            axes[i].legend(patches, labels)
    for figi in figures:
        figi.tight_layout()
    return figures, axes


def generate_2d_scatter(data, variables, class_data=None, class_names=None,
                        max_plots=6, ncol=3, sharex=False, sharey=False,
                        show_legend=True, xy_line=False, trendline=False,
                        **kwargs):
    """Generate 2D scatter plots from the given data and variables.

    This method will generate 2D scatter plots for all combinations
    of the given variables.

    Parameters
    ----------
    data : object like :class:`pandas.core.frame.DataFrame`
        The data we will plot here.
    variables : list of strings
        The variables we will generate scatter plots for.
    class_data : object like :class:`pandas.core.series.Series`, optional
        Class information for the points (if available).
    class_names : dict of strings, optional
        A mapping from the class data to labels/names.
    max_plots : integer, optional
        The maximum number of plots in a figure.
    ncol : integer, optional
        The number of columns to use in a figure.
    sharex : boolean, optional
        If True, the scatter plots will share the x-axis.
    sharey : boolean, optional
        If True, the scatter plots will share the y-axis.
    show_legend : boolean, optional
        If True, we will create a legend here and show it.
    xy_line : boolean, optional
        If True, we will add a x=y line to the plot.
    trendline : boolean, optional
        If True, we will add a trend line to the plot.
    kwargs : dict, optional
        Additional arguments used for the plotting.

    Returns
    -------
    figures : list of objects like :class:`matplotlib.figure.Figure`
        The figures containing the plots.
    axes : list of objects like :class:`matplotlib.axes.Axes`
        The axes containing the plots.

    """
    nplots = comb(len(variables), 2, exact=True)
    figures, axes = create_fig_and_axes(
        nplots, max_plots, ncol=ncol, sharex=sharex, sharey=sharey
    )
    fig = None
    for i, (xvar, yvar) in enumerate(combinations(variables, 2)):
        # We do not want to repeat the legend in all subplots:
        show_legend_ax = False
        if axes[i].figure != fig:
            fig = axes[i].figure
            show_legend_ax = True
        _, _, patches, labels = plot_scatter(
            data,
            xvar,
            yvar,
            axi=axes[i],
            class_data=class_data,
            class_names=class_names,
            **kwargs
        )
        if xy_line:
            line_xy = add_xy_line(axes[i], alpha=0.7, color='black')
            patches.append(line_xy)
            labels.append('x = y')
        if trendline:
            line_trend = add_trendline(axes[i], data[xvar], data[yvar],
                                       alpha=0.7, ls='--', color='black')
            patches.append(line_trend)
            labels.append('y = a + bx')
        if show_legend and show_legend_ax and patches and labels:
            axes[i].legend(patches, labels)
    for figi in figures:
        figi.tight_layout()
    return figures, axes


def plot_3d_scatter(data, xvar, yvar, zvar, class_data=None,
                    class_names=None, **kwargs):
    """Make a 3D scatter plot of the given data.

    Parameters
    ----------
    data : object like :class:`pandas.core.frame.DataFrame`
        The data we are plotting.
    xvar : string
        The column to use as the x-variable.
    yvar : string
        The column to use as the y-variable.
    zvar : string
        The column to use as the z-variable
    class_data : object like :class:`pandas.core.series.Series`, optional
        Class information for the points (if available).
    class_names : dict of strings, optional
        A mapping from the class data to labels/names.
    kwargs : dict, optional
        Additional arguments used for the plotting.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure containing the plot.
    axi : object like :class:`matplotlib.axes.Axes`, optional
        The axis containing the plot.

    """
    color_class, color_labels, idx_class = generate_class_colors(class_data)
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
        create_scatter_legend(
            axi, color_labels, class_names, show=True, **kwargs
        )
    fig.tight_layout()
    return fig, axi


def generate_3d_scatter(data, variables, class_data=None, class_names=None,
                        max_plots=5, **kwargs):
    """Generate 3D scatter plots from the given data and variables.

    This method will generate 3D scatter plots for all combinations
    of the given variables. Note that if the number of plots is large,
    then no plots will be generated and a warning will be issued. The
    maximum number of plots to create can be set with the parameter
    `max_plots`

    Parameters
    ----------
    data : object like :class:`pandas.core.frame.DataFrame`
        The data we will plot here.
    variables : list of strings
        The variables we will generate scatter plots for.
    class_data : object like :class:`pandas.core.series.Series`, optional
        Class information for the points (if available).
    class_names : dict of strings, optional
        A mapping from the class data to labels/names.
    max_plots : integer, optional
        The maximum number of plots to create.
    kwargs : dict, optional
        Additional arguments used for the plotting.

    Returns
    -------
    figures : list of objects like :class:`matplotlib.figure.Figure`
        The figures created here.
    axes : list of objects like :class:`matplotlib.axes.Axes`
        The axes created here.

    """
    figures = []
    axes = []
    if len(variables) < 3:
        raise ValueError(
            'For generating 3D plots, at least 3 variables must be provided.'
        )
    nplots = comb(len(variables), 3, exact=True)
    if nplots > max_plots:
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
