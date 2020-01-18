# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""psynlig - A package for creating plots with matplotlib."""
from .version import VERSION as __version__
from .colors import generate_colors
from .heatmap import plot_heatmap, annotate_heatmap, heatmap
from .scatter import plot_3d_scatter, generate_3d_scatter, plot_scatter
