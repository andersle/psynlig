# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
PCA Loadings (2D) with xkcd style and centered axes
===================================================

This example will plot PCA loadings along two principal axes.
Here we employ the
`xkcd style <https://matplotlib.org/gallery/showcase/xkcd.html>`_
and also modify the loadings plot to use centered axes
(that is, the axes go through the origin).
"""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from psynlig import pca_2d_loadings
plt.xkcd()

data_set = load_diabetes()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])
data = scale(data)

pca = PCA()
pca.fit_transform(data)

pca_2d_loadings(
    pca,
    data_set['feature_names'],
    select_components={(1, 2)},
    style='center'
)

# Remove text and add legend:
_, axes = pca_2d_loadings(
    pca,
    data_set['feature_names'],
    select_components={(1, 2)},
    style='center',
    text_settings={'show': False},
)
for axi in axes:
    axi.legend(loc='upper left')

plt.show()
