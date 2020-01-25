# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module defining common methods."""
import copy
from itertools import combinations
from math import ceil
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
from shapely.geometry import Polygon


MARKERS = {
    0: 'o',
    1: 's',
    2: 'X',
    3: 'D',
    4: 'v',
    5: '^',
    6: '<',
    7: '>',
    8: 'P',
    9: '*',
}


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
    grid : object like :class:`matplotlib.gridspec.GridSpec`
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
    figures : list of objects like :class:`matplotlib.figure.Figure`
        The figures created here.
    axes : list of objects like :class:`matplotlib.axes.Axes`
        The axes created here.

    """
    figures, axes = [], []
    nfigures = 1
    if nplots > max_plots:
        nfigures = ceil(nplots / max_plots)
    for _ in range(nfigures):
        plots = max_plots if nplots > max_plots else nplots
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
    axi : object like :class:`matplotlib.axes.Axes`
        The axis to add the x==y line to.
    **kwargs : dict, optional
        Additional arguments passed to the plotting method.

    Returns
    -------
    line : object like :class:`matplotlib.lines.Line2D`
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
    axi : object like :class:`matplotlib.axes.Axes`
        The axis to add the trendline to.
    xdata : object like :class:`pandas.core.series.Series`
        The x-values to add a trendline for.
    ydata : object like :class:`pandas.core.series.Series`
        The y-values to add a trendline for.
    **kwargs : dict, optional
        Additional arguments passed to the plotting method.

    Returns
    -------
    line : object like :class:`matplotlib.lines.Line2D`
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


def set_origin_axes(axi, xlabel, ylabel, **kwargs):
    """Move the x/y-axes of a plot to the origin.

    Parameters
    ----------
    axi : object like :class:`matplotlib.axes.Axes`
        The axis to modify.
    xlabel : string
        The label to use for the x-axis.
    ylabel : string
        The label to use for the y-axis.
    kwargs : dict, optional
        Additional font settings for the axis labels.

    """
    font_dict_x = copy.deepcopy(kwargs)
    font_dict_x.update(
        {
            'verticalalignment': 'center',
            'horizontalalignment': 'left',
        }
    )
    font_dict_y = copy.deepcopy(kwargs)
    font_dict_y.update(
        {
            'horizontalalignment': 'center',
            'verticalalignment': 'bottom',
        }
    )
    axi.spines['left'].set_position('zero')
    axi.spines['right'].set_visible(False)
    axi.spines['bottom'].set_position('zero')
    axi.spines['top'].set_visible(False)
    axi.set(xlabel=None, ylabel=None)
    axi.text(1.1, 0.0, xlabel, **font_dict_x)
    axi.text(0.0, 1.1, ylabel, **font_dict_y)


def find_axis_intersection(axi, xcoeff, ycoeff):
    """Find intersection between a line and the bounds of the axis.

    Parameters
    ----------
    axi : object like :class:`matplotlib.axes.Axes`
        The plot to add the loadings to.
    xcoeff : float
        The x-value for the line we are to extend.
    ycoeff : float
        The y-value for the line we are to extend.

    Return
    ------
    xend : float
        The x ending point for the extended line.
    yend : float
        The y ending point for the extended line.

    """
    xmin, xmax = min(axi.get_xlim()), max(axi.get_xlim())
    ymin, ymax = min(axi.get_ylim()), max(axi.get_ylim())
    xend, yend = None, None

    def direction(xhat, yhat):
        return np.sign(xcoeff * xhat + ycoeff * yhat) > 0

    if xcoeff == 0 and ycoeff == 0:
        # Can not extend it...
        pass
    else:
        if xcoeff == 0:
            xend = 0
            yend = ymax if ycoeff > 0 else ymin
        elif ycoeff == 0:
            xend = xmax if xcoeff > 0 else xmin
            yend = 0
        else:
            # Line 1)
            yhat = ycoeff * xmin / xcoeff
            if ymin <= yhat <= ymax and direction(xmin, yhat):
                xend = xmin
                yend = yhat
            # Line 2)
            xhat = xcoeff * ymin / ycoeff
            if xmin <= xhat <= xmax and direction(xhat, ymin):
                xend = xhat
                yend = ymin
            # Line 3)
            yhat = ycoeff * xmax / xcoeff
            if ymin <= yhat <= ymax and direction(xmax, yhat):
                xend = xmax
                yend = yhat
            # Line 4)
            xhat = xcoeff * ymax / ycoeff
            if xmin <= xhat <= xmax and direction(xhat, ymax):
                xend = xhat
                yend = ymax
    return xend, yend


def _get_text_boxes(axi, texts):
    """Create bounding boxes from texts.

    Parameters
    ----------
    axi : object like :class:`matplotlib.axes.Axes`
        The axis the text boxes reside in.
    texts : list of objects like :class:`matplotlib.text.Text`
        The text boxes we attempt to jiggle around.

    Returns
    -------
    boxes : list of objects like :class:`shapely.geometry.polygon.Polygon`
        The polygons of bounding boxes.

    """
    renderer = axi.figure.canvas.get_renderer()
    transform_data = axi.transData.inverted()
    boxes = []
    for txt in texts:
        box = txt.get_window_extent(renderer=renderer)
        box_data = box.transformed(transform_data)
        polygon = Polygon(
            [
                (box_data.x0, box_data.y0),
                (box_data.x0, box_data.y1),
                (box_data.x1, box_data.y1),
                (box_data.x1, box_data.y0),
            ]
        )
        boxes.append(polygon)
    return boxes


def jiggle_text(axi, texts, maxiter=1000):
    """Attempt to jiggle text around so that they do not overlap.

    Parameters
    ----------
    axi : object like :class:`matplotlib.axes.Axes`
        The axis the text boxes reside in.
    texts : list of objects like :class:`matplotlib.text.Text`
        The text boxes we attempt to jiggle around.
    maxiter : integer, optional
        The maximum number of attempts we make to jiggle the
        text around.

    """
    jiggle_x = (max(axi.get_xlim()) - min(axi.get_xlim())) * 0.01
    jiggle_y = (max(axi.get_ylim()) - min(axi.get_ylim())) * 0.01
    for _ in range(maxiter):
        boxes = _get_text_boxes(axi, texts)
        no_overlap = True
        # Check all pairs to see who overlap:
        for idx1, idx2 in combinations(range(len(boxes)), 2):
            box1 = boxes[idx1]
            box2 = boxes[idx2]
            text1 = texts[idx1]
            text2 = texts[idx2]
            if box1.intersects(box2):
                no_overlap = False
                center1 = np.array(box1.centroid)
                center2 = np.array(box2.centroid)
                dist = (center1 - center2) / norm(center1 - center2)
                vec = np.array([dist[1] * jiggle_x, dist[0] * jiggle_y])
                text1.set_va('center')
                text1.set_ha('center')
                text2.set_va('center')
                text2.set_ha('center')
                text1.set_position(center1 + vec)
                text2.set_position(center2 - vec)
                break
        if no_overlap:
            break
        # Add a white background to the text boxes:
        for txt in texts:
            txt.set_backgroundcolor('#ffffffe0')
