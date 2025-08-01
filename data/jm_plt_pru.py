# jm_utils/data/jm_matplotlib
"""
¡?
"""

## TO-DO
## pie - paretto - en lo posible mismos parámetros


__version__ = "0.1.0"
__description__ = "Custom pandas functions for data cleaning and manipulation."
__author__ = "Jorge Monti"
__email__ = "jorgitomonti@gmail.com"
__license__ = "MIT"
__status__ = "Development"
__python_requires__ = ">=3.11"
__last_modified__ = "2025-06-30"


## Standard Libs
from typing import Union, Optional, Any, Literal, Sequence, TypeAlias
import random

# Third-Party Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter  # for pareto chart and ?
import seaborn as sns

# Local Libs
from jm_utils.data.jm_pandas import to_series, get_fdt



## Custom types for non-included typing annotations
IndexElement: TypeAlias = Union[str, int, float, pd.Timestamp]
# IndexElement: TypeAlias = Union[str, int, float, 'datetime.datetime', np.str_, np.int64, np.float64, np.datetime64, pd.Timestamp, ...]


def pru1(data: list):
    sr = to_series(data)
    fig, ax = plt.subplots()
    ax.pie(x=sr)



if __name__ == "__main__":

    # Data
    dic = {'1603 SW': [21, 'No POE'], '1608 SW': [6, 'Headset compatible'], 
       '1616 SW': [3, 'Telefonista'], '9611 G': [8, 'Gerencial Gigabit']}
    df = pd.DataFrame.from_dict(dic, orient='index', columns=['Stock', 'Obs'])

    lst = list(df['Stock'])
    print(lst)

    # Pru1
    # show_plt_palettes()



