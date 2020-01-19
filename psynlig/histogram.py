# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""A module for generating histograms plots of variables."""
from .colors import generate_class_colors
from .common import create_fig_and_axes


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
    axes : list of objects like py:class:`matplotlib.axes.Axes`
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
    color_class, _, idx_class = generate_class_colors(class_data)
    fig = None
    for i, xvar in enumerate(variables):
        show_legend = False
        if axes[i].figure != fig:
            fig = axes[i].figure
            show_legend = True
        axes[i].set(xlabel=xvar, ylabel='Frequency')
        if class_data is None:
            axes[i].hist(data[xvar], **kwargs)
        else:
            for class_id, idx in idx_class.items():
                axes[i].hist(
                    data[xvar][idx],
                    color=color_class[class_id],
                    label=class_names[class_id],
                    density=False,
                    **kwargs
                )
                if show_legend:
                    axes[i].legend()
    for fig in figures:
        fig.tight_layout()
        fig.tight_layout()
    return figures, axes
