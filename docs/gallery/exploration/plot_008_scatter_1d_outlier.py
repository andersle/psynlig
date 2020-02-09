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
from tabulate import tabulate
from psynlig import generate_1d_scatter
plt.style.use('seaborn-talk')


data_set = load_diabetes()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])

variables = ['s1', 's2', 's3', 's4', 's5', 's6']

kwargs = {
    'scatter': {
        'marker': 'o',
        's': 200,
        'alpha': 0.7,
    },
    'scatter-outlier': {
        's': 100,
        'marker': 'o',
        'label': 'Outliers',
    },
    'figure': {'figsize': (12, 6)},
}

_, _, outliers = generate_1d_scatter(
    data,
    variables,
    show_legend=True,
    outliers=True,
    **kwargs,
)
for var, out in outliers.items():
    print('\nPossible outliers for "{}":\n'.format(var))
    data_out = data.iloc[out, :]
    print(
        tabulate(data_out, tablefmt='github', headers='keys', floatfmt='.2g')
    )

plt.show()
