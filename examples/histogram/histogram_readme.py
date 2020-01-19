# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""An example of generating histogram plots of raw data."""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from psynlig import histograms
plt.style.use('seaborn-talk')


data_set = load_iris()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])
class_data = data_set['target']
class_names = dict(enumerate(data_set['target_names']))
variables = ['sepal length (cm)', 'sepal width (cm)',
             'petal length (cm)', 'petal width (cm)']
figs, _ = histograms(data, variables, class_names=class_names,
                     class_data=class_data, ncol=2, max_plots=4,
                     edgecolor='black', alpha=0.8)
figs[0].savefig('histogram.png', bbox_inches='tight')
plt.show()
