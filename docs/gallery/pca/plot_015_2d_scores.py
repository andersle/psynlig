# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
PCA Scores (2D)
===============

This example will plot PCA scores along two principal axes.
"""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from psynlig import pca_2d_scores
plt.style.use('seaborn-talk')

data_set = load_breast_cancer()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])
xvars = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness',
    'mean compactness',
    'mean concavity',
    'mean concave points',
    'mean symmetry',
    'mean fractal dimension',
]
data = data[xvars]
class_data = data_set['target']
class_names = dict(enumerate(data_set['target_names']))
data = scale(data)

pca = PCA()
scores = pca.fit_transform(data)

pca_2d_scores(
    pca,
    scores,
    class_data=class_data,
    class_names=class_names,
    select_components={(1, 2), (1, 3), (2, 3)},
    s=200,
    alpha=.8
)

plt.show()
