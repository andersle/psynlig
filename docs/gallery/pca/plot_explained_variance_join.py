# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
Explained variance (pie inset)
==============================

This example will show the explained variance from a
`principal component analysis
<https://en.wikipedia.org/wiki/Principal_component_analysis>`_
as a function of the number of principal components considered.
Here we join all three plots of the explained variance into one
figure.
"""
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from psynlig import (
    pca_explained_variance,
    pca_explained_variance_bar,
    pca_explained_variance_pie
)
plt.style.use('seaborn-talk')


data_set = load_wine()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])
data = scale(data)

pca = PCA(n_components=4)
pca.fit_transform(data)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
pca_explained_variance_bar(pca, axi=ax1, alpha=0.8)
pca_explained_variance(pca, axi=ax2, marker='o', markersize=16, alpha=0.8)
fig.tight_layout()
ax3 = inset_axes(ax1, width='45%', height='45%', loc=9)
pca_explained_variance_pie(pca, axi=ax3)

plt.show()
