# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
PCA variable contributions (bubble version)
===========================================

This example will plot contributions to the principal
components from the original variables. Here, we show
the absolute values of the coefficients in a bubble heat map.
"""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from psynlig import pca_loadings_map
plt.style.use('seaborn-talk')

data_set = load_diabetes()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])
data = scale(data)

pca = PCA()
pca.fit_transform(data)

kwargs = {
    'heatmap': {
        'vmin': -1,
        'vmax': 1,
    },
}


# Plot the value of the coefficients:
pca_loadings_map(
    pca,
    data_set['feature_names'],
    bubble=True,
    annotate=False,
    **kwargs
)

# Plot the absolute value of the coefficients:
kwargs['heatmap']['vmin'] = 0
pca_loadings_map(
    pca,
    data_set['feature_names'],
    bubble=True,
    annotate=False,
    plot_style='absolute',
    **kwargs
)

plt.show()
