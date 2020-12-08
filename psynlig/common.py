# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module defining common methods."""
import copy
from itertools import combinations
from math import ceil
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
from shapely.geometry import Polygon


MARKERS = [
    'o',
    's',
    'X',
    'D',
    'v',
    '^',
    '<',
    '>',
    'P',
    '*',
    '8',
    'h',
    'H',
    '+',
    'x',
    'd',
    '1',
    '2',
    '3',
    '4',
]


GRID = {
    1: (1, 1),
    2: (1, 2),
    3: (1, 3),
    4: (2, 2),
    5: (2, 3),
    6: (3, 2),
    7: (3, 3),
    8: (2, 4),
    9: (3, 3),
    10: (2, 5),
    11: (3, 4),
    12: (3, 4),
}


DEFAULT_FIGURE = {
    'constrained_layout': True,
}


def get_figure_kwargs(kwargs):
    """Helper method to process figure kwargs."""
    fig_kw = copy.deepcopy(kwargs.get('figure', {}))
    for key, val in DEFAULT_FIGURE.items():
        if key not in fig_kw:
            fig_kw[key] = val
    return fig_kw


def set_up_fig_and_axis(fig, axi):
    """Create a figure and axis if needed.

    Parameters
    ----------
    fig : object like :class:`matplotlib.figure.Figure`
        The current figure. If None is given, we create a new one here.
    axi : object like :class:`matplotlib.axes.Axes`
        The current axis. If None is given, we create a new one here.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure created here, if any. If no figure was created,
        this is just the figure we got as a parameter.
    axi : object like :class:`matplotlib.axes.Axes`
        The axis created here. If no axis was created, this is just
        the figure we got as a parameter.

    """
    if axi is None:  # No axis, create one:
        if fig is None:  # No figure, create axis and figure:
            fig, axi = plt.subplots()
        else:
            try:
                # Check if the figure contain some axes and use the
                # first one:
                axi = fig.axes[0]
            except IndexError:
                # Could not find axes. Create one:
                axi = fig.add_subplot()
    return fig, axi


def create_fig_and_axes(nplots, nrows=None, ncols=None, **kwargs):
    """Create a set of figures and axes.

    The number of plots per figure is limited by the specified rows
    and columns. The plots will be created with constrained layout unless
    this is explicitly set to False.

    Parameters
    ----------
    nplots : integer
        The total number of plots to make.
    nrows : integer
        The number of rows to create in each plot.
    ncols : integer
        The number of columns to create in each plot.
    kwargs : dict
        Extra settings for creating the figure(s).

    Returns
    -------
    figures : list of objects like :class:`matplotlib.figure.Figure`
        The figures created here.
    axes : list of objects like :class:`matplotlib.axes.Axes`
        The axes created here.

    """
    figures, axes = [], []
    nfigures = 1
    if nrows is None or ncols is None:
        nrows, ncols = GRID.get(nplots, (4, 4))
    max_plots = nrows * ncols
    if nplots > max_plots:
        nfigures = ceil(nplots / max_plots)
    # Add constrained layout as default if it is not explicitly set
    # to false:
    fig_kw = get_figure_kwargs(kwargs)
    for _ in range(nfigures):
        # We make the same number of figures per plot as this
        # gives the same size.
        figi, axi = plt.subplots(nrows=nrows, ncols=ncols, **fig_kw)
        figures.append(figi)
        axes.extend(axi.flatten())
    # Hide axis if we created some extra ones:
    for i, axi in enumerate(axes):
        if i >= nplots:
            axi.axis('off')
    return figures, axes


def add_xy_line(axi, **kwargs):
    """Add a y=x line to the given axes.

    Parameters
    ----------
    axi : object like :class:`matplotlib.axes.Axes`
        The axis to add the y=x line to.
    **kwargs : dict, optional
        Additional arguments passed to the plotting method.

    Returns
    -------
    line : object like :class:`matplotlib.lines.Line2D`
        The created y=x line.

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
    """Obtain the coefficient of determination (R^2).

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
    """Move the x and y-axes of a plot to the origin.

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
    """Find intersection between a line and the axis bounds.

    Parameters
    ----------
    axi : object like :class:`matplotlib.axes.Axes`
        The axis we will find intersections in,
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
            # Possibility 1:
            yhat = ycoeff * xmin / xcoeff
            if ymin <= yhat <= ymax and direction(xmin, yhat):
                xend = xmin
                yend = yhat
            # Possibility 2:
            xhat = xcoeff * ymin / ycoeff
            if xmin <= xhat <= xmax and direction(xhat, ymin):
                xend = xhat
                yend = ymin
            # Possibility 3:
            yhat = ycoeff * xmax / xcoeff
            if ymin <= yhat <= ymax and direction(xmax, yhat):
                xend = xmax
                yend = yhat
            # Possibility 4:
            xhat = xcoeff * ymax / ycoeff
            if xmin <= xhat <= xmax and direction(xhat, ymax):
                xend = xhat
                yend = ymax
    return xend, yend


def _get_text_boxes(axi, texts):
    """Get bounding boxes for the givens text elements.

    Parameters
    ----------
    axi : object like :class:`matplotlib.axes.Axes`
        The axis the text boxes reside in.
    texts : list of objects like :class:`matplotlib.text.Text`
        The text boxes we attempt to jiggle around.

    Returns
    -------
    boxes : list of objects like :class:`shapely.geometry.polygon.Polygon`
        Polygons representing the bounding boxes.

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


def get_selector(components, select_components, combi):
    """Get a selector for components.

    This is helper method in case we select a subset of
    components, or wish to plot for all combinations.

    Parameters
    ----------
    components : integer
        The number of components we are selecting from,
    select_components : iterable or None
        The items we are to pick. If this is None, we select
        all combinations.
    combi : integer
        The number of combinations of the components we
        are selecting, in the case we are to generate them here.

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


def iqr_outlier(data, variables):
    """Locate outliers by computing the interquartile range.

    Parameters
    ----------

    Returns
    -------
    out_of_bounds : object like :class:`pandas.core.frame.DataFrame`
    outliers : dict of integer
        For each variable, these are the indexes of possible outliers.
    (upper, lower) : tuple of objects like :class:`pandas.core.series.Series`
        These are the bounds for outlier detection.

    """
    sub_data = data[variables]
    quant1 = sub_data.quantile(0.25)
    quant3 = sub_data.quantile(0.75)
    iqr = quant3 - quant1
    lower = quant1 - 1.5 * iqr
    upper = quant3 + 1.5 * iqr
    out_of_bounds = (
        (sub_data[variables] < lower) | (sub_data[variables] > upper)
    )
    # Convert to indexes to help with 1D plotting:
    outliers = {}
    for vari in variables:
        outliers[vari] = out_of_bounds[out_of_bounds[vari]].index.values
    return out_of_bounds, outliers, (upper, lower)


def get_text_settings(settings, default=None):
    """Get text settings for loadings.

    Parameters
    ----------
    settings : dict or None
        The provided settings.
    default : dict or None,
        The default settings. In case None is given, we use
        hard-coded default settings given here.

    Returns
    -------
    text_settings : dict
        A dict containing the text settings.
    outline_settings : dict
        A dict containing settings for creating a stroke outline.

    """
    outline_settings = {}
    if default is None:
        text_settings = {
            'weight': 'bold',
            'horizontalalignment': 'left',
            'verticalalignment': 'center',
            'fontsize': 'large',
        }
    else:
        text_settings = copy.deepcopy(default)

    if settings is None:
        # Just return the defaults.
        return text_settings, outline_settings

    if settings:
        text_settings.update(settings)

    if 'outline' in text_settings:
        outline_settings = {'linewidth': 1, 'foreground': 'black'}
        outline_settings.update(text_settings.get('outline', {}))
        del text_settings['outline']
    return text_settings, outline_settings
