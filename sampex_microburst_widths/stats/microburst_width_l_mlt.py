# Make a plot of the microburst width as a function of L and MLT
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config
from sampex_microburst_widths.misc import plot_annotator_decorator
from sampex_microburst_widths.stats import dial_plot

catalog_name = 'microburst_catalog_01.csv'

@plot_annotator_decorator.annotate_plot
def main():
    r2_thresh = 0.9
    max_width = 0.25
    # width_bins=np.linspace(0, max_width, num=50)
    L_bins = np.linspace(2, 8.1, num=20)
    MLT_bins = np.linspace(0, 24, num=40)

    df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', catalog_name))
    df.dropna(inplace=True)
    df = df[df['width_s'] < max_width]
    df['fwhm'] = df['fwhm'].abs()
    df = df[df.r2 > r2_thresh]

    num_microbursts_H, _, _ = np.histogram2d(df['MLT'], df['L_Shell'],
                                            bins=[MLT_bins, L_bins])
    width_H = np.nan*np.zeros((len(MLT_bins), len(L_bins)))

    for i, (start_MLT, end_MLT) in enumerate(zip(MLT_bins[:-1], MLT_bins[1:])):
        for j, (start_L, end_L) in enumerate(zip(L_bins[:-1], L_bins[1:])):
            df_flt = df.loc[(
                (df['MLT'] > start_MLT) &  (df['MLT'] < end_MLT) &
                (df['L_Shell'] > start_L) &  (df['L_Shell'] < end_L)
                ), 'fwhm']
            if df_flt.shape[0] > 50:
                width_H[i, j] = df_flt.median()

    fig = plt.figure(figsize=(10, 4))
    ax = plt.subplot(121, projection='polar')
    bx = plt.subplot(122, projection='polar')

    d = dial_plot.Dial(ax, MLT_bins, L_bins, width_H)
    d.draw_dial(L_labels=[2,4,6],
                colorbar_kwargs={'label':'Median microburst width [s]'})
    
    d2 = dial_plot.Dial(bx, MLT_bins, L_bins, num_microbursts_H)
    d2.draw_dial(L_labels=[2,4,6],
                colorbar_kwargs={'label':'Number of microbursts'})

    plt.suptitle('SAMPEX microburst width statistics')
    return

main()
plt.show()