# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module defining helper methods for creating a heat map."""
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np


def heatmap(data, row_labels, col_labels, axi=None, fig=None,
            cbar_kw=None, cbarlabel='', **kwargs):
    """Create a heat map from a numpy array and two lists of labels.

    Parameters
    ----------
    data : object like :py:class:`numpy.ndarray`
        A 2D numpy array of shape (N, M).
    row_labels : list of strings
        A list or array of length N with the labels for the rows.
    col_labels : list of strings
        A list or array of length M with the labels for the columns.
    axi : object like :py:class:`matplotlib.axes.Axes`, optional
        An axis to plot the heat map.  If not provided, a new axis
        will be created.
    fig : object like :py:class:`matplotlib.figure.Figure`, optional
        The figure where the axes resides in. If given, tight layout
        will be applied.
    cbar_kw : dict, optional
        A dictionary with arguments to the creation of the color bar.
    cbarlabel : string, optional
        The label for the color bar.
    **kwargs : dict, optional
        Arguments used for drawing the heat map.

    Returns
    -------
    fig : object like :py:class:`matplotlib.figure.Figure`
        The figure in which the heatmap is plotted.
    axi : object like :py:class:`matplotlib.axes.Axes`
        The axis to which the heatmap is added.
    img : object like :py:class:`matplotlib.image.AxesImage`
        The generated heat map.
    cbar : object like :py:class:`matplotlib.colorbar.Colorbar`
        The color bar created for the heat map.

    """
    if axi is None:
        if fig is None:
            fig, axi = plt.subplots()
        else:
            try:
                axi = fig.axes[0]
            except IndexError:
                # Could not find axes. Create one:
                axi = fig.add_subplot()

    # Plot the heatmap:
    img = axi.imshow(data, **kwargs)

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

    # Let the horizontal axes labeling appear on top:
    axi.tick_params(
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False
    )

    # Turn spines off:
    for _, spine in axi.spines.items():
        spine.set_visible(False)

    # Add white grid:
    axi.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    axi.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
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
    img : object like :py:class:`matplotlib.image.AxesImage`
        The heat map image to be labeled.
    data : object like :py:class:`numpy.ndarray`, optional
        Data used to annotate the heat map. If not given, the
        data in the image is used.
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


def plot_heatmap(data, val_fmt='{x:.2f}', textcolors=None, **kwargs):
    """Plot a heat map to investigate correlations.

    Parameters
    ----------
    data : object like :py:class:`pandas.DataFrame`
        The data we will generate a heat correlation map from.
    val_fmt : string, optional
        The format of the annotations inside the heat map.
    textcolors : list of strings, optional
        Colors used for the text. The number of colors provided defines
        a binning for the data values, and values are colored with the
        corresponding color. If no colors are provided, all are colored
        black.
    **kwargs : dict, optional
        Arguments used for drawing the heat map.

    Returns
    -------
    fig : object like :py:class:`matplotlib.figure.Figure`
        The figure in which the heatmap is plotted.
    ax1 : object like :py:class:`matplotlib.axes.Axes`
        The axis to which the heat map is added.

    """
    corr = data.corr(method='pearson')
    fig1, ax1 = plt.subplots()
    _, _, img, _ = heatmap(
        corr,
        data.columns,
        data.columns,
        axi=ax1,
        cbarlabel='Pearson correlation coefficient',
        **kwargs.get('heatmap', {}),
    )
    annotate_heatmap(
        img,
        val_fmt=val_fmt,
        textcolors=textcolors,
        **kwargs.get('text', {}),
    )
    ax1.grid(False)
    fig1.tight_layout()
    return fig1, ax1
