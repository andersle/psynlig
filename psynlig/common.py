# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module defining common methods."""
from math import ceil
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import pearsonr


def create_grid(n_plots, ncol):
    """Create a grid for matplotlib given a number of plots and columns.

    Parameters
    ----------
    nplots : integer
        The number of plots to create in total.
    ncol : integer
        The number of columns to create.

    Returns
    -------
    grid : object like :py:class:`matplotlib.gridspec.GridSpec`
        The grid we will create axes in.

    """
    nrow = ceil(n_plots / ncol)
    grid = GridSpec(nrow, ncol)
    return grid


def create_fig_and_axes(nplots, max_plots, ncol=3, sharex=False, sharey=False):
    """Create a set of figures and axes.

    The number of plots per figure is limited to the given parameter
    ``max_plots``.

    Parameters
    ----------
    nplots : integer
        The total number of plots to make.
    max_plots : integer
        The maximum number of plots in a figure.
    ncol : integer
        The number of columns to create in each plot.
    sharex : boolean
        If True, the axes will share the x-axis.
    sharey : boolean
        If True, the axes will share the y-axis.

    Returns
    -------
    figures : list of objects like :py:class:`matplotlib.figure.Figure`
        The figures created here.
    axes : list of objects like :py:class:`matplotlib.axes.Axes`
        The axes created here.

    """
    figures, axes = [], []
    nfigures = 1
    if nplots > max_plots:
        nfigures = ceil(nplots / max_plots)
    for _ in range(nfigures):
        if nplots > max_plots:
            plots = max_plots
        else:
            plots = nplots
        grid = create_grid(plots, ncol)
        nplots -= plots
        figi = plt.figure()
        figures.append(figi)
        ax0 = None
        for j in range(plots):
            row, col = divmod(j, ncol)
            if ax0 is None:
                ax0 = axi = figi.add_subplot(grid[row, col])
            else:
                if sharex and sharey:
                    axi = figi.add_subplot(grid[row, col],
                                           sharex=ax0, sharey=ax0)
                else:
                    if sharex:
                        axi = figi.add_subplot(grid[row, col], sharex=ax0)
                    elif sharey:
                        axi = figi.add_subplot(grid[row, col], sharey=ax0)
                    else:
                        axi = figi.add_subplot(grid[row, col])
            axes.append(axi)
    return figures, axes


def add_xy_line(axi, **kwargs):
    """Add a x==y line to the given axes.

    Parameters
    ----------
    axi : object like :py:class:`matplotlib.axes.Axes`
        The axis to add the x==y line to.
    **kwargs : dict, optional
        Additional arguments passed to the plotting method.

    Returns
    -------
    line : object like :py:class:`matplotlib.lines.Line2D`
        The created line.

    """
    lim_min = np.min([axi.get_xlim(), axi.get_ylim()])
    lim_max = np.max([axi.get_xlim(), axi.get_ylim()])
    line, = axi.plot([lim_min, lim_max], [lim_min, lim_max], **kwargs)
    return line


def add_trendline(axi, xdata, ydata, **kwargs):
    """Add a trendline to the given axes.

    Parameters
    ----------
    axi : object like :py:class:`matplotlib.axes.Axes`
        The axis to add the trendline to.
    xdata : object like :py:class:`pandas.core.series.Series`
        The x-values to add a trendline for.
    ydata : object like :py:class:`pandas.core.series.Series`
        The y-values to add a trendline for.
    **kwargs : dict, optional
        Additional arguments passed to the plotting method.

    Returns
    -------
    line : object like :py:class:`matplotlib.lines.Line2D`
        The created line.

    """
    param = np.polyfit(xdata, ydata, 1)
    yhat = np.polyval(param, xdata)
    rsq = get_rsquared(ydata, yhat)
    corr = pearsonr(xdata, ydata)
    text = r'R$^2$ = {:.2f}, $\rho$ = {:.2f}'.format(rsq, corr[0])
    xpoint = np.array([min(xdata), max(xdata)])
    line, = axi.plot(xpoint, np.polyval(param, xpoint), **kwargs)
    axi.set_title(text)
    return line


def get_rsquared(yval, yre):
    """Obtain R^2 for fitted data.

    Parameters
    ----------
    yval : numpy.array
        The y-values used in the fitting.
    yre : numpy.array
        The estimated y-values from the fitting.

    Returns
    -------
    rsq : float
        The estimated value of R^2.

    Notes
    -----
    https://en.wikipedia.org/wiki/Coefficient_of_determination

    """
    ss_tot = np.sum((yval - yval.mean())**2)
    ss_res = np.sum((yval - yre)**2)
    rsq = 1.0 - (ss_res / ss_tot)
    return rsq
