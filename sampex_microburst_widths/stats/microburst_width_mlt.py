# Histogram the microburst widths as a function of MLT
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config
from sampex_microburst_widths.misc import plot_annotator_decorator

@plot_annotator_decorator.annotate_plot
def main():
    max_width=0.25
    df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', 'microburst_catalog_00.csv'))
    df = df[df['width_s'] < max_width]

    _, ax = plt.subplots()
    df.plot(x='MLT', y='width_s', kind='hexbin', ax=ax)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, max_width)
    ax.text(0.95, 0.95, "O'Brien burst parameter n_100=0.1 ms, a_500=1 s"
                        "\nWidth at half prominence\nno visual inspection", 
            ha='right', va='top', transform=ax.transAxes
            )
    # MLT_bins = np.arange(0, 25, 3)
    # MLT_binned = [ 
    #             df.loc[(df['MLT'] > l_MLT) & (df['MLT'] < u_MLT), 'width_s'].to_numpy()
    #             for l_MLT, u_MLT in zip(MLT_bins[:-1], MLT_bins[1:])
    #             ]

    # _, bx = plt.subplots()
    # bx.boxplot(MLT_binned)
    return df

df = main()
plt.show()