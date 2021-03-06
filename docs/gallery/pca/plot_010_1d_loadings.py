# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
PCA Loadings (1D)
=================

This example will plot PCA loadings along a principal axis.
"""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from psynlig import pca_1d_loadings
plt.style.use('seaborn-talk')

data_set = load_diabetes()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])
data = scale(data)

pca = PCA()
pca.fit_transform(data)

pca_1d_loadings(
    pca,
    data_set['feature_names'],
    select_components={2},
    text_settings={
        'fontsize': 'x-large',
        'weight': 'bold',
    },
)

plt.show()
