# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
PCA Loadings (2D)
=================

This example will plot PCA loadings along two principal axes.
"""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from psynlig import pca_2d_loadings
plt.style.use('seaborn-talk')

data_set = load_diabetes()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])
data = scale(data)

pca = PCA()
pca.fit_transform(data)

text_settings = {
    'fontsize': 'xx-large',
    'outline': {'foreground': '0.2'}
}

pca_2d_loadings(
    pca,
    data_set['feature_names'],
    select_components={(3, 4)},
    text_settings=text_settings
)

plt.show()
