# Histogram the microburst widths as a function of AE
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config
from sampex_microburst_widths.misc import plot_annotator_decorator

@plot_annotator_decorator.annotate_plot
def main():
    max_width=0.25
    width_bins=np.linspace(0, max_width, num=20)
    ae_bins = [0, 100, 300, 1000]

    r2_thresh = 0.9
    df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', 'microburst_catalog_02.csv'))
    df.dropna(inplace=True)

    if hasattr(df, 'fwhm'):
        df['fwhm'] = df['fwhm'].abs()
        # Filter by the R^2 value with the microburst FWHM 
        df = df[(df['adj_r2'] > r2_thresh) & (df['fwhm'] < max_width)]
    else:
        df = df[df['width_s'] < max_width]

    _, ax = plt.subplots()

    for start_ae, end_ae in zip(ae_bins[:-1], ae_bins[1:]):
        df_flt = df[(df['AE'] > start_ae) & (df['AE'] < end_ae)]
        ax.hist(df_flt['fwhm'], bins=width_bins, histtype='step', density=True,
                label=f'{start_ae} < AE < {end_ae}')
    
    # df['fwhm'].plot.hist(ax=ax, bins=np.linspace(0, max_width))
    # ax.text(0.95, 0.95, "O'Brien burst parameter n_100=0.1 s, a_500=0.5 s"
    #                     f"\nGaussian FWHM | min R^2 = {r2_thresh}"
    #                     "\nno visual inspection", 
    #         ha='right', va='top', transform=ax.transAxes
    #         )
    return df

df = main()
plt.legend()
plt.show()