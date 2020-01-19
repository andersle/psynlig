# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""
Generating 3D scatter plots of raw data
=======================================

This example uses the method :py:meth:`psynlig.scatter.plot_3d_scatter`
for explorative plotting of raw data using three variables
(i.e. for generating 3D plots). The data points can be colored
according to class labels if this is available.
This is done by passing the labels for each data point (using
the parameter ``class_data``) and a mapping from the labels
to something more human-readable (using the parameter ``class_names``).

Generating several plots (for all combinations of 3 variables) can
be done using the method :py:meth:`psynlig.scatter.generate_3d_scatter`.
Note that this can create a large number of plots and that the
method will only generate these if the number of plots does not
exceed the parameter ``max_plots`` of this method.
"""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from psynlig import plot_3d_scatter
plt.style.use('seaborn-talk')
data_set = load_iris()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])
class_data = data_set['target']
class_names = dict(enumerate(data_set['target_names']))
plot_3d_scatter(data, 'sepal length (cm)', 'sepal width (cm)',
                'petal length (cm)', class_names=class_names,
                class_data=class_data, marker='o', s=200, alpha=0.7)
plt.show()
