# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
Styling a heat map of correlations
==================================

This is an example of styling a heat map for showing correlations
between variables.
The correlation between variables is obtained as the
`Pearson correlation
coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_.

"""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine as load_data
from psynlig import plot_heatmap
plt.style.use('ggplot')


data_set = load_data()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])

kwargs = {
    'text': {
        'fontsize': 'large',
    },
    'heatmap': {
        'vmin': -1,
        'vmax': 1,
        'cmap': 'magma',
    }
}

fig, _ = plot_heatmap(data, textcolors=['white', 'black'], **kwargs)
# Increase figure size:
fig.set_size_inches(14, 10)
fig.tight_layout()
plt.show()
