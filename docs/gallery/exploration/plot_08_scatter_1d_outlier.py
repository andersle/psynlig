# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
r"""
Generating 1D scatter plots with outliers
=========================================

This examle will plot the raw data points in a 1D scatter plot.
Here, we attempt to highlight outliers by calculating the
interquartile range.
"""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from psynlig import generate_1d_scatter
plt.style.use('seaborn-talk')


data_set = load_diabetes()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])

variables = ['s1', 's2', 's3', 's4', 's5', 's6']

generate_1d_scatter(
    data,
    variables,
    ncol=2,
    show_legend=True,
    outliers=True,
    marker='o',
    s=200,
    alpha=0.7
)

plt.show()
