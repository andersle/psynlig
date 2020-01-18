# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""An example of generating a heat map of correlations."""
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from psynlig import plot_heatmap
plt.style.use('seaborn-talk')


def load_iris_set():
    """Set up the data set."""
    data_set = load_iris()
    data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])
    data['target'] = data_set['target']
    xvars = [
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ]
    xdata = data[xvars]
    return data, xdata


def main():
    """Generate the heat map."""
    _, xdata = load_iris_set()
    plot_heatmap(xdata, vmin=-1, vmax=1)
    plot_heatmap(xdata, textcolors=['white', 'black'], vmin=-1, vmax=1)
    plot_heatmap(xdata, textcolors=['white', 'black'])
    plot_heatmap(xdata, textcolors=['white'], cmap='Spectral')
    plt.show()


if __name__ == '__main__':
    main()
