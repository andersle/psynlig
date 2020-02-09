# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
PCA Scores (3D)
===============

This example will plot PCA scores along three principal axes.
"""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from psynlig import pca_3d_scores
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

loading_settings = {
    'adjust_text': False,
    'add_text': True,
    'jiggle_text': True,
}

pca_3d_scores(
    pca,
    scores,
    class_data=class_data,
    class_names=class_names,
    select_components={(1, 2, 3)},
    s=200,
    alpha=.8,
    cmap_classes='Dark2',
)

plt.show()
