# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""An example of generating a heat map of correlations."""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine as load_data
from psynlig import plot_heatmap
plt.style.use('seaborn-talk')


data_set = load_data()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])
plot_heatmap(data, vmin=-1, vmax=1, textcolors=['white', 'black'],
             cmap='viridis')
plt.show()
