# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""psynlig - A package for creating plots with matplotlib."""
from .version import VERSION as __version__
from .colors import generate_colors
from .heatmap import plot_heatmap, annotate_heatmap, heatmap
from .scatter import (
    plot_3d_scatter,
    generate_3d_scatter,
    plot_scatter,
    generate_scatter,
)
from .histogram import histograms
from .pca import (
    pca_explained_variance,
    pca_explained_variance_bar,
    pca_explained_variance_pie,
    pca_1d_loadings,
    pca_1d_loadings_component,
    pca_2d_loadings,
    pca_2d_loadings_component,
    pca_3d_loadings,
    pca_3d_loadings_component,
    pca_2d_scores,
    pca_1d_scores,
)
