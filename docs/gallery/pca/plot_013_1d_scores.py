# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
PCA Scores (1D)
===============

This example will plot PCA scores along one principal axis.
"""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from psynlig import pca_1d_scores
plt.style.use('seaborn-talk')

data_set = load_breast_cancer()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])
class_data = data_set['target']
class_names = dict(enumerate(data_set['target_names']))
data = scale(data)

pca = PCA()
reduced_features = pca.fit_transform(data)

pca_1d_scores(
    pca,
    reduced_features,
    class_data=class_data,
    class_names=class_names,
    select_components={1, 2},
    s=200,
    alpha=.8
)

plt.show()
