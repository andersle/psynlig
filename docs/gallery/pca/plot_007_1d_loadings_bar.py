# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
PCA variable contributions
==========================

This example will plot contributions to a principal component from
the original variable.
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

for plot_type in ('bar', 'bar-square', 'bar-absolute'):
    pca_1d_loadings(
        pca,
        data_set['feature_names'],
        select_components={2},
        plot_type=plot_type,
    )

plt.show()
