# Histogram the microburst widths as a function of L shell
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config
from sampex_microburst_widths.misc import plot_annotator_decorator

@plot_annotator_decorator.annotate_plot
def main():
    r2_thresh = 0.9
    max_width = 0.25
    width_bins=np.linspace(0, max_width, num=50)
    L_bins = np.linspace(3, 8, num=50)
    LL, WW = np.meshgrid(L_bins, width_bins)

    df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', 'microburst_catalog_02.csv'))
    # df = df[df['width_s'] < max_width]
    df['fwhm'] = df['fwhm'].abs()
    df = df[df.adj_r2 > r2_thresh]
    H, _, _ = np.histogram2d(df['L_Shell'], df['fwhm'], bins=[L_bins, width_bins])
    fig, ax = plt.subplots()
    im = ax.pcolormesh(LL, WW, H.T)
    fig.colorbar(im, ax=ax)
    ax.text(0.95, 0.95, "O'Brien burst parameter n_100=0.1 s, a_500=1 s"
                        f"\nGaussian FWHM | min R^2 = {r2_thresh}"
                        "\nno visual inspection", 
            ha='right', va='top', transform=ax.transAxes, color='white'
            )
    ax.set(xlabel='L_Shell', ylabel='Width [s]')
    return df

df = main()
plt.show()