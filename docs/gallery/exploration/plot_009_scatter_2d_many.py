# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
r"""
Generating 2D scatter plots with many variables
===============================================

This example uses the method
:py:meth:`psynlig.scatter.generate_2d_scatter`
for generating a set of 2D scatter plots of combinations of variables.
In this example, we consider a case where there are many variables.
"""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from psynlig import generate_2d_scatter
plt.style.use('seaborn-talk')


data_set = load_wine()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])
class_data = data_set['target']
class_names = dict(enumerate(data_set['target_names']))

# Get variable names, but shorten them:
variables = data.columns

kwargs = {
    'scatter': {
        'marker': 'o',
        's': 75,
        'alpha': 0.7,
    },
    'figure': {'figsize': (12, 14)},
}

generate_2d_scatter(
    data,
    variables,
    class_names=class_names,
    class_data=class_data,
    ncols=5,
    nrows=6,
    show_legend=False,
    xy_line=False,
    trendline=False,
    cmap_class='viridis',
    shorten_variables=True,
    **kwargs,
)

plt.show()
