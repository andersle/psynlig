# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""An example of generating scatter plots of raw data."""
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