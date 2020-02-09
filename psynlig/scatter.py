# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module for generating scatter plots of variables."""
from itertools import combinations
import warnings
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
from scipy.special import comb
from .colors import generate_class_colors, generate_colors
from .common import (
    add_xy_line,
    add_trendline,
    create_fig_and_axes,
    iqr_outlier,
    get_figure_kwargs,
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


def plot_scatter(data, xvar, yvar, axi=None, xlabel=None, ylabel=None,
                 class_data=None, class_names=None, highlight=None,
                 cmap_class=None, **kwargs):
    """Make a 2D scatter plot of the given data.

    Parameters
    ----------
    data : object like :class:`pandas.core.frame.DataFrame`
        The data we are plotting.
    xvar : string
        The column to use as the x-variable.
    yvar : string
        The column to use as the y-variable.
    xlabel : string, optional
        The label to use for the x-axis. If None, we will use xvar.
    ylabel : string, optional
        The label to use for the y-axis. If None, we will use yvar.
    axi : object like :class:`matplotlib.axes.Axes`, optional
        An axis to add the plot to. If this is not provided,
        a new axis (and figure) will be created here.
    class_data : object like :class:`pandas.core.series.Series`, optional
        Class information for the points (if available).
    class_names : dict of strings
        A mapping from the class data to labels/names.
    highlight : list of integers, optional
        This can be used to highlight certain points in the plot.
    cmap_class : string or object like :class:`matplotlib.colors.Colormap`, optional
        A color map to use for classes.
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
    color_class, color_labels, idx_class = generate_class_colors(
        class_data, cmap=cmap_class
    )
    fig = None
    if axi is None:
        fig_kw = get_figure_kwargs(kwargs)
        fig, axi = plt.subplots(**fig_kw)
    if xvar is None:
        axi.set(xlabel='Data point no.', ylabel=yvar)
        xdata = np.arange(len(data[yvar]))
    else:
        xlabel = xvar if xlabel is None else xlabel
        ylabel = yvar if ylabel is None else ylabel
        axi.set(xlabel=xlabel, ylabel=ylabel)
        xdata = data[xvar]
    ydata = data[yvar]

    if class_data is None:
        axi.scatter(xdata, ydata, **kwargs.get('scatter', {}))
    else:
        for class_id, idx in idx_class.items():
            axi.scatter(
                xdata[idx],
                ydata[idx],
                color=color_class[class_id],
                **kwargs.get('scatter', {}),
            )
        patches, labels = create_scatter_legend(
            axi, color_labels, class_names, **kwargs.get('scatter', {}),
        )
    if highlight is not None:
        scat = axi.scatter(
            xdata[highlight],
            ydata[highlight],
            **kwargs.get('scatter-outlier', {}),
        )
        patches.append(scat)
        labels.append(scat.get_label())
    return fig, axi, patches, labels


def generate_1d_scatter(data, variables, class_data=None,
                        class_names=None, nrows=None, ncols=None,
                        sharex=False, sharey=False, show_legend=True,
                        outliers=False, cmap_class=None, **kwargs):
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
    nrows : integer, optional
        The number of rows to use in a figure.
    ncols : integer, optional
        The number of columns to use in a figure.
    sharex : boolean, optional
        If True, the scatter plots will share the x-axis.
    sharey : boolean, optional
        If True, the scatter plots will share the y-axis.
    show_legend : boolean, optional
        If True, we will create a legend here and show it.
    outliers : boolean, optional
        If True, we will try to mark outliers in the plot.
    cmap_class : string or object like :class:`matplotlib.colors.Colormap`, optional
        A color map to use for classes.
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
        nplots, nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey,
        **kwargs,
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

        _, _, patches, labels = plot_scatter(
            data,
            None,
            yvar,
            axi=axes[i],
            class_data=class_data,
            class_names=class_names,
            highlight=highlight,
            cmap_class=cmap_class,
            **kwargs,
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
    return figures, axes, outlier_points


def generate_2d_scatter(data, variables, class_data=None, class_names=None,
                        nrows=None, ncols=None, sharex=False, sharey=False,
                        show_legend=True, xy_line=False, trendline=False,
                        cmap_class=None, shorten_variables=False,
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
    nrows : integer, optional
        The number of rows to use in a figure.
    ncols : integer, optional
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
    cmap_class : string or object like :class:`matplotlib.colors.Colormap`, optional
        A color map to use for classes.
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
        nplots, nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey,
        **kwargs,
    )
    fig = None
    for i, (xvar, yvar) in enumerate(combinations(variables, 2)):
        # We do not want to repeat the legend in all subplots:
        show_legend_ax = False
        if axes[i].figure != fig:
            fig = axes[i].figure
            show_legend_ax = True
        xlabel = None
        ylabel = None
        if shorten_variables:
            if len(xvar) > 5:
                xlabel = xvar[:3] + '...'
            if len(yvar) > 5:
                ylabel = yvar[:3] + '...'
        _, _, patches, labels = plot_scatter(
            data,
            xvar,
            yvar,
            axi=axes[i],
            xlabel=xlabel,
            ylabel=ylabel,
            class_data=class_data,
            class_names=class_names,
            cmap_class=cmap_class,
            **kwargs,
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
    return figures, axes


def plot_3d_scatter(data, xvar, yvar, zvar, class_data=None,
                    class_names=None, cmap_class=None, **kwargs):
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
    cmap_class : string or object like :class:`matplotlib.colors.Colormap`, optional
        A color map to use for classes.
    kwargs : dict, optional
        Additional arguments used for the plotting.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure containing the plot.
    axi : object like :class:`matplotlib.axes.Axes`, optional
        The axis containing the plot.

    """
    color_class, color_labels, idx_class = generate_class_colors(
        class_data, cmap=cmap_class
    )
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


def scatter_1d_flat(data, class_data=None, class_names=None, scaler=None,
                    add_average=False, add_lines=False,
                    cmap_lines=None, cmap_class=None, split_class=False,
                    scatter_settings=None, line_settings=None):
    """Make a flat plot of several variables.

    Here, the points on the x-axis are the variables, while
    the y-values are points for each data series.

    Parameters
    ----------
    data : object like :class:`pandas.core.frame.DataFrame`
        The data we are plotting.
    class_data : object like :class:`pandas.core.series.Series`, optional
        Class information for the points (if available).
    class_names : dict of strings
        A mapping from the class data to labels/names.
    scaler : callable, optional
        A function that can be used to scale the variables.
    add_average : boolean, optional
        If True, we will show the averages for each variable.
    add_lines : boolean, optional
        If True, we will show lines for each "measurement".
    cmap_lines : string or object like :class:`matplotlib.colors.Colormap`, optional
        A color map to use for lines.
    cmap_class : string or object like :class:`matplotlib.colors.Colormap`, optional
        A color map to use for classes.
    split_class : boolean, optional
        If True, the plot with class information will be split into
        one plot for each class.
    scatter_settings : dict, optional
        Additional settings for the scatter plot.
    line_settings : dict, optional
        Additional settings for plotting lines.

    Returns
    -------
    figures : objects like :class:`matplotlib.figure.Figure`
        The figure created here.
    axes : object(s) like :class:`matplotlib.axes.Axes`
        The axes created here.

    """
    if class_data is None:
        return _scatter_1d_flat_no_class(data, scaler=scaler,
                                         add_average=add_average,
                                         add_lines=add_lines,
                                         cmap_lines=cmap_lines,
                                         line_settings=line_settings,
                                         scatter_settings=scatter_settings)
    return _scatter_1d_flat_class(data, class_data,
                                  split_class=split_class,
                                  class_names=class_names,
                                  scaler=scaler,
                                  cmap_class=cmap_class,
                                  add_lines=add_lines,
                                  add_average=add_average,
                                  line_settings=line_settings,
                                  scatter_settings=scatter_settings)


def _get_settings_if_empty(settings):
    """Get settings if None are given."""
    if settings is None:
        return {}
    return settings


def _scatter_1d_flat_no_class(data, scaler=None, add_average=False,
                              add_lines=False, cmap_lines=None,
                              scatter_settings=None,
                              line_settings=None):
    """Make a flat plot of several variables.

    Here, the points on the x-axis are the variables, while
    the y-values are points for each data series.

    Parameters
    ----------
    data : object like :class:`pandas.core.frame.DataFrame`
        The data we are plotting.
    scaler : callable, optional
        A function that can be used to scale the variables.
    add_average : boolean, optional
        If True, we will show the averages for each variable.
    add_lines : boolean, optional
        If True, we will show lines for each "measurement".
    cmap_lines : string or object like :class:`matplotlib.colors.Colormap`, optional
        A color map to use for lines.
    scatter_settings : dict, optional
        Additional settings for the scatter plot.
    line_settings : dict, optional
        Additional settings for plotting lines.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure containing the plot.
    axi : object like :class:`matplotlib.axes.Axes`
        The axis containing the plot.

    """
    fig, axi = plt.subplots(constrained_layout=True)
    variables = data.columns
    axi.set_xticks(range(len(variables)))
    axi.set_xticklabels(variables, rotation='vertical')
    yvalues = []
    xvalues = []
    if scaler is not None:
        axi.set_ylabel('Scaled values')
    else:
        axi.set_ylabel('Values')
    for i, variable in enumerate(variables):
        yval = data[variable]
        if scaler is not None:
            yval = scaler(yval)
        yvalues.append(yval)
        xvalues.append(np.full_like(yval, i))
    yvalues = np.array(yvalues)
    xvalues = np.array(xvalues)

    line_kw = _get_settings_if_empty(line_settings)

    if add_lines:
        colors = generate_colors(len(yvalues[0, :]), cmap=cmap_lines)
        lines = axi.plot(xvalues, yvalues, zorder=1, **line_kw)
        for line, color in zip(lines, colors):
            line.set_color(color)

    scatter_kw = _get_settings_if_empty(scatter_settings)
    axi.scatter(xvalues, yvalues, zorder=2, **scatter_kw)
    if add_average:
        avg = np.average(yvalues, axis=1)
        scat = axi.scatter(range(len(avg)), avg, zorder=3, marker='X')
        axi.plot(range(len(avg)), avg, color=scat.get_facecolors()[0])
    return fig, axi


def _scatter_1d_flat_class(data, class_data, class_names=None,
                           scaler=None,
                           add_lines=False, add_average=False,
                           cmap_class=None, split_class=False,
                           scatter_settings=None,
                           line_settings=None):
    """Make a flat plot of several variables.

    Here, the points on the x-axis are the variables, while
    the y-values are points for each data series.
    The class information is used for coloring.

    Parameters
    ----------
    data : object like :class:`pandas.core.frame.DataFrame`
        The data we are plotting.
    class_data : object like :class:`pandas.core.series.Series`, optional
        Class information for the points (if available).
    class_names : dict of strings
        A mapping from the class data to labels/names.
    scalar : callable, optional
        A function that can be used to scale the variables.
    add_average : boolean, optional
        If True, we will show the averages for each variable.
    add_lines : boolean, optional
        If True, we will show lines for each "measurement".
    cmap_class : string or object like :class:`matplotlib.colors.Colormap`, optional
        A color map to use for classes.
    split_class : boolean, optional
        If True, the plot with class information will be split into
        one plot for each class.
    scatter_settings : dict, optional
        Additional settings for the scatter plot.
    line_settings : dict, optional
        Additional settings for plotting lines.

    Returns
    -------
    figures : objects like :class:`matplotlib.figure.Figure`
        The figure created here.
    axes : object(s) like :class:`matplotlib.axes.Axes`
        The axes created here.

    """
    color_class, color_labels, idx_class = generate_class_colors(
        class_data, cmap=cmap_class
    )

    if split_class:
        fig, axes = plt.subplots(
            nrows=1, ncols=len(idx_class), constrained_layout=True,
            sharex=True, sharey=True,
        )
        all_axes = axes.flatten()
        axes = {class_id: all_axes[i] for i, class_id in enumerate(idx_class)}
    else:
        fig, axi = plt.subplots(constrained_layout=True)
        all_axes = [axi]
        axes = {class_id: axi for class_id in idx_class}

    variables = data.columns
    for _, axi in axes.items():
        axi.set_xticks(range(len(variables)))
        axi.set_xticklabels(variables, rotation='vertical')
    yvalues = {i: [] for i in idx_class}
    xvalues = {i: [] for i in idx_class}
    for i, variable in enumerate(variables):
        yval = data[variable]
        if scaler is not None:
            yval = scaler(yval)
        for class_id, idx in idx_class.items():
            yvali = yval[idx]
            xpos = np.full_like(yvali, i)
            yvalues[class_id].append(yvali)
            xvalues[class_id].append(xpos)

    line_kw = _get_settings_if_empty(line_settings)
    if add_lines:
        for class_id in idx_class:
            axes[class_id].plot(
                xvalues[class_id],
                yvalues[class_id],
                color=color_class[class_id],
                zorder=1,
                **line_kw,
            )

    scatter_kw = _get_settings_if_empty(scatter_settings)
    for class_id in idx_class:
        axes[class_id].scatter(
            xvalues[class_id],
            yvalues[class_id],
            color=color_class[class_id],
            zorder=2,
            **scatter_kw,
        )
    if add_average:
        for class_id in idx_class:
            avg = np.average(yvalues[class_id], axis=1)
            axes[class_id].scatter(
                range(len(avg)), avg, zorder=3, marker='X', color=color_class[class_id],
                edgecolor='black',
            )
    create_scatter_legend(
        all_axes[0], color_labels, class_names, show=True, **scatter_kw
    )
    return fig, all_axes
