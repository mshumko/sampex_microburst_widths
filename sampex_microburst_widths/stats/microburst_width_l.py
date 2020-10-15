# Histogram the microburst widths as a function of L shell
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config
from sampex_microburst_widths.misc import plot_annotator_decorator

@plot_annotator_decorator.annotate_plot
def main():
    width_bins=np.linspace(0, 0.50, num=20)
    L_bins = np.arange(2, 11, 0.5)
    LL, WW = np.meshgrid(L_bins, width_bins)

    df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', 'microburst_catalog_00.csv'))
    # df = df[df['width_s'] < max_width]
    H, _, _ = np.histogram2d(df['L_Shell'], df['width_s'], bins=[L_bins, width_bins])
    fig, ax = plt.subplots()
    im = ax.pcolormesh(LL, WW, H.T)
    fig.colorbar(im, ax=ax)
    ax.text(0.95, 0.95, "O'Brien burst parameter n_100=0.1 ms, a_500=1 s"
                        "\nWidth at half prominence\nno visual inspection", 
            ha='right', va='top', transform=ax.transAxes, color='white'
            )
    ax.set(xlabel='L_Shell', ylabel='Width [s]')
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