# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
Explained variance (pie chart)
==============================

This example will show the explained variance from a
`principal component analysis
<https://en.wikipedia.org/wiki/Principal_component_analysis>`_
in a pie chart.
"""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from psynlig import pca_explained_variance_pie
plt.style.use('seaborn-talk')


data_set = load_wine()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])
data = scale(data)

pca = PCA(n_components=5)
pca.fit_transform(data)

fig, axi = pca_explained_variance_pie(pca)
axi.set_title('Explained variance by principal components')

plt.show()
