# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
Generating a heat map of correlations (bubble version)
======================================================

This is an example of generating a heat map for showing correlations
between variables.
The correlation between variables is obtained as the
`Pearson correlation
coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_.

The heat map is generated from a :class:`pandas.core.frame.DataFrame`
and all pairs of variables (based on columns) are considered.

Here, we show a version where the correlation values are shown as
circles where the radius is proportional to the correlation.

"""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine as load_data
from psynlig import plot_correlation_heatmap
plt.style.use('seaborn-talk')


data_set = load_data()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])

kwargs = {
    'heatmap': {
        'vmin': -1,
        'vmax': 1,
        'cmap': 'viridis',
    }
}

fig, _ = plot_correlation_heatmap(data, bubble=True, annotate=False, **kwargs)
# Increase figure size:
fig.set_size_inches(14, 10)
fig.tight_layout()
plt.show()
