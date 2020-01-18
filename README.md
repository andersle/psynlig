# psynlig

A library for creating plots with matplotlib.

## Examples

### Generating a heat map for correlations
The following code
```python
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from psynlig import plot_heatmap
plt.style.use('seaborn-talk')


data_set = load_iris()
data = pd.DataFrame(data_set['data'], columns=data_set['feature_names'])
fig, _ = plot_heatmap(data, vmin=-1, vmax=1, textcolors=['white', 'black'])
fig.savefig('heatmap_example.png', bbox_inches='tight')
plt.show()
```

will generate the following figure:

![heatmap](examples/heat_correlation/heatmap_example.png)

