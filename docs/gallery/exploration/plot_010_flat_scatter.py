# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
r"""
Generating 1D scatter plots with many variables
===============================================

This example will plot observations for several variables
in a 1D plot.
"""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import scale
from psynlig import scatter_1d_flat
plt.style.use('seaborn-talk')


data_set = load_wine()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])
class_data = data_set['target']
class_names = dict(enumerate(data_set['target_names']))

scatter_settings = {'alpha': 0.5, 's': 100}
line_settings = {'alpha': 0.5}

_, axi = scatter_1d_flat(
    data,
    scaler=scale,
    add_lines=True,
    scatter_settings=scatter_settings,
    line_settings=line_settings
)
axi.set_title('Scaled variables, individual colors:')

_, axi = scatter_1d_flat(
    data,
    scaler=None,
    add_average=True,
    scatter_settings=scatter_settings,
    line_settings=line_settings
)
axi.set_title('Unscaled variables, without lines:')

_, axes = scatter_1d_flat(
    data,
    class_data=class_data,
    class_names=class_names,
    scaler=scale,
    add_lines=True,
    add_average=True,
    scatter_settings=scatter_settings,
    line_settings=line_settings
)
axes[0].set_title('Scaled variables, using class information:')

fig, _ = scatter_1d_flat(
    data,
    class_data=class_data,
    class_names=class_names,
    scaler=scale,
    add_lines=True,
    split_class=True,
    add_average=True,
    scatter_settings=scatter_settings,
    line_settings=line_settings
)
fig.suptitle('Scaled variables, using class information for splitting:')


plt.show()
