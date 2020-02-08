# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module defining helper methods for creating a heat map."""
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.patches import Circle, Rectangle
import numpy as np
from .common import set_up_fig_and_axis, get_figure_kwargs


def create_bubbles(data, img, axi):
    """Create bubbles for a heat map.

    Parameters
    ----------
    data : object like :class:`numpy.ndarray`
        A 2D numpy array of shape (N, M).
    img : object like :class:`matplotlib.image.AxesImage`
        The heat map image we have generated.
    axi : object like :class:`matplotlib.axes.Axes`
        The axis to add the bubbles to.

    """
    vals = img.get_array()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = img.norm(vals[i, j])
            radius = np.abs(vals[i, j]) * 0.5 * 0.9
            color = img.cmap(value)
            if i % 2 == 0:
                rect = Rectangle((j-0.5, i-0.5), 1, 1, color='0.8')
            else:
                rect = Rectangle((j-0.5, i-0.5), 1, 1, color='0.9')
            axi.add_artist(rect)
            circle = Circle((j, i), radius=radius,
                            color=color)
            axi.add_artist(circle)
    img.set_visible(False)


def heatmap(data, row_labels, col_labels, axi=None, fig=None,
            cbar_kw=None, cbarlabel='', bubble=False, **kwargs):
    """Create a heat map from a numpy array and two lists of labels.

    Parameters
    ----------
    data : object like :class:`numpy.ndarray`
        A 2D numpy array of shape (N, M).
    row_labels : list of strings
        A list or array of length N with the labels for the rows.
    col_labels : list of strings
        A list or array of length M with the labels for the columns.
    axi : object like :class:`matplotlib.axes.Axes`, optional
        An axis to plot the heat map. If not provided, a new axis
        will be created.
    fig : object like :class:`matplotlib.figure.Figure`, optional
        The figure where the axes resides in. If given, tight layout
        will be applied.
    cbar_kw : dict, optional
        A dictionary with arguments to the creation of the color bar.
    cbarlabel : string, optional
        The label for the color bar.
    bubble : boolean, optional
        If True, we will draw bubbles indicating the size
        of the given data points.
    **kwargs : dict, optional
        Additional arguments for drawing the heat map.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure in which the heatmap is plotted.
    axi : object like :class:`matplotlib.axes.Axes`
        The axis to which the heatmap is added.
    img : object like :class:`matplotlib.image.AxesImage`
        The generated heat map.
    cbar : object like :class:`matplotlib.colorbar.Colorbar`
        The color bar created for the heat map.

    """
    fig, axi = set_up_fig_and_axis(fig, axi)

    # Plot the heatmap:
    img = axi.imshow(data, **kwargs)
    # Check if this is a bubble map:
    if bubble:
        create_bubbles(data, img, axi)

    # Create colorbars:
    if cbar_kw is None:
        cbar_kw = {}
    cbar = axi.figure.colorbar(img, ax=axi, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va='bottom')

    # Show ticks using the provided labels:
    axi.set_xticks(np.arange(data.shape[1]))
    axi.set_xticklabels(
        col_labels,
        rotation=-30,
        horizontalalignment='right',
        rotation_mode='anchor'
    )
    axi.set_yticks(np.arange(data.shape[0]))
    axi.set_yticklabels(row_labels)

    # Labels on top:
    axi.tick_params(
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False
    )

    # Hide spines off:
    for _, spine in axi.spines.items():
        spine.set_visible(False)

    # Add grid:
    axi.grid(False)
    axi.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    axi.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    if bubble:
        axi.grid(which='minor', color='white', linestyle='-', linewidth=3)
        axi.tick_params(which='minor', bottom=False, left=False)
        axi.tick_params(which='major', top=False, left=False)
    else:
        axi.grid(which='minor', color='white', linestyle='-', linewidth=3)
        axi.tick_params(which='minor', bottom=False, left=False)
    if fig is not None:
        fig.tight_layout()
    return fig, axi, img, cbar


def annotate_heatmap(img, data=None, val_fmt='{x:.2f}', textcolors=None,
                     **kwargs):
    """Annotate a heatmap with values.

    Parameters
    ----------
    img : object like :class:`matplotlib.image.AxesImage`
        The heat map image to be labeled.
    data : object like :class:`numpy.ndarray`, optional
        Data used to annotate the heat map. If not given, the
        data in the heat map image (``img``) is used.
    val_fmt : string, optional
        The format of the annotations inside the heat map.
    textcolors : list of strings, optional
        Colors used for the text. The number of colors provided defines
        a binning for the data values, and values are colored with the
        corresponding color. If no colors are provided, all are colored
        black.
    **kwargs : dict, optional
        Extra arguments used for creating text labels.

    """
    if data is None:
        data = img.get_array()

    # Create arguments for text:
    textkw = kwargs.copy()
    textkw.update(
        {
            'horizontalalignment': 'center',
            'verticalalignment': 'center',
        }
    )

    # Get the formatter:
    formatter = StrMethodFormatter(val_fmt)

    if textcolors is None:
        textcolors = ['black']

    texts = []
    bins = np.linspace(0, 1, len(textcolors) + 1)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = img.norm(data[i, j])
            idx = np.digitize(val, bins, right=True)
            idx = max(idx - 1, 0)
            textkw.update(color=textcolors[idx])
            text = img.axes.text(j, i, formatter(data[i, j], None), **textkw)
            texts.append(text)
    return texts


def plot_correlation_heatmap(data, val_fmt='{x:.2f}', bubble=False,
                             annotate=True, textcolors=None, **kwargs):
    """Plot a heat map to investigate correlations.

    Parameters
    ----------
    data : object like :class:`pandas.DataFrame`
        The data we will generate a heat correlation map from.
    val_fmt : string, optional
        The format of the annotations inside the heat map.
    bubble : optional, boolean
        If True, we will draw bubbles to indicate the size of the
        given data points.
    annotate : boolean, optional
        If True, we will annotate the plot with values.
    textcolors : list of strings, optional
        Colors used for the text. The number of colors provided defines
        a binning for the data values, and values are colored with the
        corresponding color. If no colors are provided, all are colored
        black.
    **kwargs : dict, optional
        Arguments used for drawing the heat map.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure in which the heatmap is plotted.
    ax1 : object like :class:`matplotlib.axes.Axes`
        The axis to which the heat map is added.

    """
    corr = data.corr(method='pearson')
    fig1, ax1 = plot_annotated_heatmap(
        corr,
        data.columns,
        data.columns,
        cbarlabel='Pearson correlation coefficient',
        val_fmt=val_fmt,
        bubble=bubble,
        textcolors=textcolors,
        annotate=annotate,
        **kwargs
    )
    return fig1, ax1


def plot_annotated_heatmap(data, row_labels, col_labels, cbarlabel='',
                           val_fmt='{x:.2f}', textcolors=None, bubble=False,
                           annotate=True, **kwargs):
    """Plot a heat map to investigate correlations.

    Parameters
    ----------
    data : object like :class:`numpy.ndarray`
        A 2D numpy array of shape (N, M).
    row_labels : list of strings
        A list or array of length N with the labels for the rows.
    col_labels : list of strings
        A list or array of length M with the labels for the columns.
    cbarlabel : string, optional
        The label for the color bar.
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
        If True, we will annotate the plot with values.
    **kwargs : dict, optional
        Arguments used for drawing the heat map.

    Returns
    -------
    fig : object like :class:`matplotlib.figure.Figure`
        The figure in which the heatmap is plotted.
    ax1 : object like :class:`matplotlib.axes.Axes`
        The axis to which the heat map is added.

    """
    fig_kw = get_figure_kwargs(kwargs)
    fig1, ax1 = plt.subplots(**fig_kw)
    _, _, img, _ = heatmap(
        data,
        row_labels,
        col_labels,
        axi=ax1,
        cbarlabel=cbarlabel,
        bubble=bubble,
        **kwargs.get('heatmap', {}),
    )
    if annotate:
        annotate_heatmap(
            img,
            val_fmt=val_fmt,
            textcolors=textcolors,
            **kwargs.get('text', {}),
        )
    return fig1, ax1
