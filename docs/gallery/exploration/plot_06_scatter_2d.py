# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
r"""
Generating 2D scatter plots
===========================

This example uses the method
:py:meth:`psynlig.scatter.generate_2d_scatter`
for generating a set of 2D scatter plots of combinations of variables.
This is intended for investigating possible correlations visually
between pairs of variables.

The data points can be colored
according to class labels if this is available.
This is done by passing the labels for each data point (using
the parameter ``class_data``) and a mapping from the labels
to something more human-readable (using the parameter ``class_names``).

A `trend line <https://en.wikipedia.org/wiki/Linear_regression>`_ is added
to the plot and the calculated
`coefficient of determination
<https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
is shown in the plot (as :math:`R^2`). In addition, the calculated
`Pearson correlation
coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_
is also shown (:math:`\rho`).
"""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from psynlig import generate_2d_scatter
plt.style.use('seaborn-talk')


data_set = load_iris()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])
class_data = data_set['target']
class_names = dict(enumerate(data_set['target_names']))

variables = ['sepal length (cm)', 'sepal width (cm)',
             'petal length (cm)', 'petal width (cm)']

generate_2d_scatter(
    data,
    variables,
    class_names=class_names,
    class_data=class_data,
    show_legend=True,
    xy_line=True,
    trendline=True,
    marker='o',
    s=200,
    alpha=0.7
)

plt.show()
